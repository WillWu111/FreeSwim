from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn
from functools import partial, lru_cache
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
create_block_mask = torch.compile(create_block_mask)


BLOCK_MASK = None
BLOCK_MASK_ = None


@lru_cache
def init_mask_flex(num_frames, height, width, d_h, d_w, device):
    
    def get_mask(b, h, q_idx, kv_idx):
        q_t = q_idx // (height * width)
        q_hw = q_idx % (height * width)
        q_h = q_hw // width
        q_w = q_hw % width
        d_b = max(d_h - q_h, 0)
        d_u = max(d_h + q_h - height + 1, 0)
        d_r = max(d_w - q_w, 0)
        d_l = max(d_w + q_w - width + 1, 0)

        kv_t = kv_idx // (height * width)
        kv_hw = kv_idx % (height * width)
        kv_h = kv_hw // width
        kv_w = kv_hw % width

        return torch.logical_and(
            torch.logical_and(
                kv_h <= q_h + d_h + d_b,
                kv_h >= q_h - d_h - d_u
            ),
            torch.logical_and(
                kv_w <= q_w + d_w + d_r,
                kv_w >= q_w - d_w - d_l
            )
        )
    
    global BLOCK_MASK
    BLOCK_MASK = create_block_mask(get_mask, B=None, H=None, Q_LEN=num_frames * height * width, 
                                   KV_LEN=num_frames * height * width, device=device, _compile=True)

                     
def _get_qkv_projections(attn, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn, encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanFlexAttnProcessor_:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        assert BLOCK_MASK is not None
        self.flex_attn = partial(flex_attention, block_mask=BLOCK_MASK)
        self.flex_attn = torch.compile(self.flex_attn, dynamic=False)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        reuse_cross_output: Optional[torch.Tensor] = None,
        cache_cross_ouput: Optional[bool] = False,
        steps: Optional[int] = None,
        cond: Optional[bool] = False,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        attention_scale = math.sqrt(math.log(30 * 52 * 2, 30 * 52) / key.size(3))

        # 这里先算一遍local的self
        hidden_states = self.flex_attn(
            query[0:1].transpose(1, 2),
            key[0:1].transpose(1, 2),
            value[0:1].transpose(1, 2),
            scale=attention_scale
        ).transpose(1, 2)

        # 然后再算一遍global的self
        if cond:
            if cache_cross_ouput:
                print("cond的需要计算global的self")
                hidden_states_ = dispatch_attention_fn(
                    query[1:2],
                    key[1:2],
                    value[1:2],
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                    scale=attention_scale
                )
            else:
                print("cond的不需要计算global的self")
                hidden_states_ = hidden_states
        else:
            print("uncond的self不需要global分支")
            hidden_states_ = hidden_states

        # (B, L, C) -> (B * 2, L, C)
        hidden_states = torch.cat([hidden_states, hidden_states_], dim=0)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states



class WanCrossAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        reuse_cross_output: Optional[torch.Tensor] = None,
        cache_cross_ouput: Optional[bool] = False,
        steps: Optional[int] = None,
        cond: Optional[bool] = False,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if cond:
            if cache_cross_ouput:
                print("step",steps,"cond的需要缓存计算global的cross attn")
                key = torch.cat([key, key.clone()], dim=0)
                value = torch.cat([value, value.clone()], dim=0)

                hidden_states = dispatch_attention_fn(
                    query[1:2],
                    key[1:2],
                    value[1:2],
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                )
                cached_cross_output = hidden_states # (1, L, C)
                hidden_states = torch.cat([hidden_states, hidden_states.clone()], dim=0)
                # print("hidden_states",hidden_states.shape)
                
            else:
                print("step",steps,"cond的不需要缓存计算global的cross attn")
                hidden_states = torch.cat([reuse_cross_output, reuse_cross_output.clone()], dim=0)
        else:
            print("uncond的cross不需要global分支")
            hidden_states = dispatch_attention_fn(
                query[0:1],
                key[0:1],
                value[0:1],
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            # print("hidden_states",hidden_states.shape) # (1, L, C)
            hidden_states = torch.cat([hidden_states, hidden_states.clone()], dim=0)

   
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if cond and cache_cross_ouput:
            return hidden_states, cached_cross_output
        else:
            return hidden_states, None
