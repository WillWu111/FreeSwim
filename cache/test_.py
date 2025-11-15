import cv2
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video, load_video
# from pipeline_wan_video2video import WanVideoToVideoPipeline
from pipeline_wan import WanPipeline
from transformer_wan import WanTransformer3DModel
from pipeline_wan_video2video import WanVideoToVideoPipeline

from attention_processor_ import init_mask_flex, WanFlexAttnProcessor_, WanCrossAttnProcessor


model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer",torch_dtype=torch.bfloat16) 
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

height = 1088
width = 1920
num_frames = 81
base_height = 480
base_width = 832
prompt = "A realistic close-up of an elderly man with gray hair and a thick gray beard, wearing a light-colored shirt. His head is slightly lowered. The camera zooms from full body to close-up, highlighting detailed facial wrinkles, skin texture, forehead lines, eye bags, and beard strands. High resolution, cinematic lighting, sharp details."
negative_prompt = "repeating patterns, Blurry face, low detail, distorted features, extra limbs, cartoon style, smooth plastic skin, low resolution, flat colors, lack of texture"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=5.0
).frames[0]
# export_to_video(output, "base_video.mp4", fps=15)
pipe = WanVideoToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer",torch_dtype=torch.bfloat16) 
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

init_mask_flex(
    num_frames=1 + (num_frames - 1) // 4, 
    height=height // 16, 
    width=width // 16,
    d_h=base_height // 16 // 2, 
    d_w=base_width // 16 // 2, 
    device='cuda'
)

attn_processors = {}
for k in pipe.transformer.attn_processors.keys():
    if 'attn2' in k:
        attn_processors[k] = WanCrossAttnProcessor()
    else:
        attn_processors[k] = WanFlexAttnProcessor_()
pipe.transformer.set_attn_processor(attn_processors)

pipe.scheduler.config.flow_shift = 9.0

output = [cv2.resize(item, (width, height), interpolation=cv2.INTER_LINEAR) for item in output]

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=5.0,
)

export_to_video(output, "Old_man.mp4", fps=15)
