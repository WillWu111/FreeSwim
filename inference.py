import cv2
import torch
import argparse
import cv2 
import torch 
from diffusers import AutoencoderKLWan, WanPipeline 
from diffusers.utils import export_to_video, load_video

parser = argparse.ArgumentParser(description='Choose between cache and nocache')
parser.add_argument('--mode', type=str, choices=['cache', 'nocache'], default='nocache', help="Specify either 'cache' or 'nocache' mode")
parser.add_argument('--target_height', type=int, default=1088, help="Target height for Your Super Resolution Video")
parser.add_argument('--target_width', type=int, default=1920, help="Target width for Your Super Resolution Video")
args = parser.parse_args()



if args.mode == 'cache':
    from cache.pipeline_wan import WanPipeline
    from cache.transformer_wan import WanTransformer3DModel
    from cache.pipeline_wan_video2video import WanVideoToVideoPipeline
    from cache.attention_processor_ import init_mask_flex, WanFlexAttnProcessor_, WanCrossAttnProcessor
elif args.mode == 'nocache':
    from nocache.pipeline_wan import WanPipeline
    from nocache.transformer_wan import WanTransformer3DModel
    from nocache.pipeline_wan_video2video import WanVideoToVideoPipeline
    from nocache.attention_processor_ import init_mask_flex, WanFlexAttnProcessor_, WanCrossAttnProcessor

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer",torch_dtype=torch.bfloat16) 
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

num_frames = 81
base_height = 480
base_width = 832
prompt = "A realistic close-up of an elderly man with gray hair and a thick gray beard, wearing a light-colored shirt. His head is slightly lowered. The camera zooms from full body to close-up, highlighting detailed facial wrinkles, skin texture, forehead lines, eye bags, and beard strands. High resolution, cinematic lighting, sharp details."
negative_prompt = "repeating patterns, Blurry face, low detail, distorted features, extra limbs, cartoon style, smooth plastic skin, low resolution, flat colors, lack of texture"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=base_height,
    width=base_width,
    num_frames=num_frames,
    guidance_scale=5.0
).frames[0]

pipe = WanVideoToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16) 
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

init_mask_flex(
    num_frames=1 + (num_frames - 1) // 4, 
    height=args.target_height // 16, 
    width=args.target_width // 16,
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

output = [cv2.resize(item, (args.target_width, args.target_height), interpolation=cv2.INTER_LINEAR) for item in output]

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=args.target_height,
    width=args.target_width,
    num_frames=num_frames,
    guidance_scale=5.0,
)

export_to_video(output, "Old_man.mp4", fps=15)
