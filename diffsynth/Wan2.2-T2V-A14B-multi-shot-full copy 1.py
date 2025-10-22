import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline, ModelConfig

device='cuda'
pipe = WanVideoHoloCinePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        ModelConfig(path="/robby/share/Editing/hf_weights/Wan-AI/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(path="/robby/share/Editing/yihao/outputs/multi_shot_wan/diffsynth/Wan2.2-T2V-A14B_15s_test_cross_attn_high_noise/step-3000/model_load.safetensors", offload_device="cpu"),
        ModelConfig(path="/robby/share/Editing/yihao/outputs/multi_shot_wan/diffsynth/Wan2.2-T2V-A14B_15s_test_cross_attn_low_noise/step-3000/model_load.safetensors",  offload_device="cpu"),
        ModelConfig(path="/robby/share/Editing/hf_weights/Wan-AI/Wan2.2-T2V-A14B/Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

pipe.to(device)

# Text-to-video
video = pipe(
    prompt="[global caption] The scene features a young painter, [character1], with paint-smudged cheeks and intense, focused eyes. Her hair is tied up messily. The setting is a bright, sun-drenched art studio with large windows, canvases, and the smell of oil paint. This scene contains 6 shots. [per shot caption] Medium shot of [character1] standing back from a large canvas, brush in hand, critically observing her work. [shot cut] Close-up of her hand holding the brush, dabbing it thoughtfully onto a palette of vibrant colors. [shot cut] Extreme close-up of her eyes, narrowed in concentration as she studies the canvas. [shot cut] Close-up on the canvas, showing a detailed, textured brushstroke being slowly applied. [shot cut] Medium close-up of [character1]'s face, a small, satisfied smile appears as she finds the right color. [shot cut] Over-the-shoulder shot showing her add a final, delicate highlight to the painting.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    shot_cut_frames=[37, 73, 113, 169, 205],
    height=480,
    width=832,
    num_frames=241,
    num_inference_steps=50,
)
save_video(video, "/ossfs/workspace/test_multi_shot_full.mp4", fps=15, quality=5)
