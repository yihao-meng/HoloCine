import torch
import math
from diffsynth import save_video
from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline, ModelConfig

# ---------------------------------------------------
#                Helper Functions
# ---------------------------------------------------

def enforce_4t_plus_1(n: int) -> int:
    """Forces an integer 'n' to the closest 4t+1 form."""
    t = round((n - 1) / 4)
    return 4 * t + 1

def prepare_multishot_inputs(
    global_caption: str,
    shot_captions: list[str],
    total_frames: int,
    custom_shot_cut_frames: list[int] = None
) -> dict:

    
    num_shots = len(shot_captions)
    
    # 1. Prepare 'prompt'
    if "This scene contains" not in global_caption:
        global_caption = global_caption.strip() + f" This scene contains {num_shots} shots."
    per_shot_string = " [shot cut] ".join(shot_captions)
    prompt = f"[global caption] {global_caption} [per shot caption] {per_shot_string}"

    # 2. Prepare 'num_frames'
    processed_total_frames = enforce_4t_plus_1(total_frames)

    # 3. Prepare 'shot_cut_frames'
    num_cuts = num_shots - 1
    processed_shot_cuts = []

    if custom_shot_cut_frames:
        # User provided custom cuts
        print(f"Using {len(custom_shot_cut_frames)} user-defined shot cuts (enforcing 4t+1).")
        for frame in custom_shot_cut_frames:
            processed_shot_cuts.append(enforce_4t_plus_1(frame))
    else:
        # Auto-calculate cuts
        print(f"Auto-calculating {num_cuts} shot cuts.")
        if num_cuts > 0:
            ideal_step = processed_total_frames / num_shots
            for i in range(1, num_shots):
                approx_cut_frame = i * ideal_step
                processed_shot_cuts.append(enforce_4t_plus_1(round(approx_cut_frame)))

    processed_shot_cuts = sorted(list(set(processed_shot_cuts)))
    processed_shot_cuts = [f for f in processed_shot_cuts if f > 0 and f < processed_total_frames]

    return {
        "prompt": prompt,
        "shot_cut_frames": processed_shot_cuts,
        "num_frames": processed_total_frames
    }

# ---------------------------------------------------
# 
#           ✨ Main Inference Wrapper ✨
#
# ---------------------------------------------------

def run_inference(
    pipe: WanVideoHoloCinePipeline,
    output_path: str,
    
    # --- Prompting Options (Auto-detect) ---
    global_caption: str = None,
    shot_captions: list[str] = None,
    prompt: str = None,
    negative_prompt: str = None,
    
    # --- Core Generation Parameters (All Optional) ---
    num_frames: int = None,
    shot_cut_frames: list[int] = None,
    
    # --- Other Generation Parameters ---
    seed: int = 0,
    tiled: bool = True,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 50,
    
    # --- Output Parameters ---
    fps: int = 15,
    quality: int = 5
):
    """
    Runs the inference pipeline, auto-detecting the input mode
    and honoring pipeline defaults for optional parameters.
    
    Mode 1 (Structured): Provide 'global_caption', 'shot_captions'
                        
    Mode 2 (Raw): Provide a complete 'prompt', sturctured as 'global caption' xxx. [per shot caption]...[shot cut]...[shot cut]...
     
    """
    
    # --- 1. Prepare 'pipe_kwargs' dictionary ---
    pipe_kwargs = {
        "negative_prompt": negative_prompt,
        "seed": seed,
        "tiled": tiled,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps
    }

    # --- 2. Auto-Detection Logic ---
    if global_caption and shot_captions:
        # --- Mode 1: Structured Input ---
        print("--- Detected Structured Input (Mode 1) ---")
        if num_frames is None:
            raise ValueError("Must provide 'num_frames' for structured input (Mode 1).")
        
        # Use the helper function
        inputs = prepare_multishot_inputs(
            global_caption=global_caption,
            shot_captions=shot_captions,
            total_frames=num_frames,
            custom_shot_cut_frames=shot_cut_frames
        )
        pipe_kwargs.update(inputs)

    elif prompt:
        # --- Mode 2: Raw String Input ---
        print("--- Detected Raw String Input (Mode 2) ---")
        pipe_kwargs["prompt"] = prompt
        
        # Process num_frames ONLY if provided
        if num_frames is not None:
            processed_frames = enforce_4t_plus_1(num_frames)
            if num_frames != processed_frames:
                print(f"Corrected 'num_frames': {num_frames} -> {processed_frames}")
            pipe_kwargs["num_frames"] = processed_frames
        else:
            print("Using default 'num_frames' (if any).")
            pipe_kwargs["num_frames"] = None
        
        # Process shot_cut_frames ONLY if provided
        if shot_cut_frames is not None:
            processed_cuts = [enforce_4t_plus_1(f) for f in shot_cut_frames]
            if shot_cut_frames != processed_cuts:
                print(f"Corrected 'shot_cut_frames': {shot_cut_frames} -> {processed_cuts}")
            pipe_kwargs["shot_cut_frames"] = processed_cuts
        else:
            print("Using default 'shot_cut_frames' (if any).")
            pipe_kwargs["shot_cut_frames"] = None
        
    else:
        raise ValueError("Invalid inputs. Provide either (global_caption, shot_captions, num_frames) OR (prompt).")

    # --- 3. Filter out None values before calling pipe ---
    # This ensures we don't pass 'num_frames=None' and override a 
    # default value (e.g., num_frames=25) inside the pipeline.
    final_pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if v is not None}
    
    if "prompt" not in final_pipe_kwargs:
         raise ValueError("A 'prompt' or ('global_caption' + 'shot_captions') is required.")

    # --- 4. Run Generation ---
    print(f"Running inference...")
    if "num_frames" in final_pipe_kwargs:
        print(f"  Total frames: {final_pipe_kwargs['num_frames']}")
    if "shot_cut_frames" in final_pipe_kwargs:
        print(f"  Cuts: {final_pipe_kwargs['shot_cut_frames']}")

    video = pipe(**final_pipe_kwargs)
    
    save_video(video, output_path, fps=fps, quality=quality)
    print(f"Video saved successfully to {output_path}")


# ---------------------------------------------------
# 
#                 Script Execution
#
# ---------------------------------------------------

# --- 1. Load Model (Done once) ---
device = 'cuda'
pipe = WanVideoHoloCinePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        ModelConfig(path="/robby/share/Editing/hf_weights/Wan-AI/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(path="/robby/share/Editing/yihao/outputs/multi_shot_wan/diffsynth/Wan2.2-T2V-A14B_15s_test_cross_attn_high_noise/step-3000/model_load.safensors", offload_device="cpu"),
        ModelConfig(path="/robby/share/Editing/yihao/outputs/multi_shot_wan/diffsynth/Wan2.2-T2V-A14B_15s_test_cross_attn_low_noise/step-3000/model_load.safensors",  offload_device="cpu"),
        ModelConfig(path="/robby/share/Editing/hf_weights/Wan-AI/Wan2.2-T2V-A14B/Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()
pipe.to(device)

# --- 2. Define Common Parameters ---
scene_negative_prompt = "vibrant, overexposed, static, blurry details, subtitles, style, painting, picture, still, grayscale, worst quality, low quality, jpeg artifacts, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static image, cluttered background, three legs, crowded background, walking backwards"


# ===================================================================
#                ✨ How to Use ✨
# ===================================================================

# --- Example 1: Call using Structured Input (Mode 1) ---
# (Auto-calculates shot cuts)
print("\n--- Running Example 1 (Structured Input) ---")
run_inference(
    pipe=pipe,
    negative_prompt=scene_negative_prompt,
    output_path="/ossfs/workspace/test_structured_output.mp4",
    
    # Mode 1 inputs
    global_caption="The scene features a young painter, [character1], with paint-smudged cheeks and intense, focused eyes. Her hair is tied up messily. The setting is a bright, sun-drenched art studio.",
    shot_captions=[
        "Medium shot of [character1] standing back from a large canvas, brush in hand, critically observing her work.",
        "Close-up of her hand holding the brush, dabbing it thoughtfully onto a palette of vibrant colors.",
        "Extreme close-up of her eyes, narrowed in concentration as she studies the canvas.",
        "Close-up on the canvas, showing a detailed, textured brushstroke being slowly applied.",
        "Medium close-up of [character1]'s face, a small, satisfied smile appears as she finds the right color.",
        "Over-the-shoulder shot showing her add a final, delicate highlight to the painting."
    ],
    
    # 'num_frames' is required for Mode 1
    num_frames=241,
    # 'shot_cut_frames' is optional (it will be auto-calculated)
    shot_cut_frames=[37, 73, 113, 169, 206] 
)


# # --- Example 2: Call using Raw String Input (Mode 2) ---
# # (Uses your original prompt format)
# print("\n--- Running Example 2 (Raw String Input) ---")
# run_inference(
#     pipe=pipe,
#     negative_prompt=scene_negative_prompt,
#     output_path="/ossfs/workspace/test_raw_string_output.mp4",
    
#     # Mode 2 inputs
#     prompt="[global caption] The scene features a young painter, [character1], with paint-smudged cheeks and intense, focused eyes. Her hair is tied up messily. The setting is a bright, sun-drenched art studio with large windows, canvases, and the smell of oil paint. This scene contains 6 shots. [per shot caption] Medium shot of [character1] standing back from a large canvas, brush in hand, critically observing her work. [shot cut] Close-up of her hand holding the brush, dabbing it thoughtfully onto a palette of vibrant colors. [shot cut] Extreme close-up of her eyes, narrowed in concentration as she studies the canvas. [shot cut] Close-up on the canvas, showing a detailed, textured brushstroke being slowly applied. [shot cut] Medium close-up of [character1]'s face, a small, satisfied smile appears as she finds the right color. [shot cut] Over-the-shoulder shot showing her add a final, delicate highlight to the painting.",
    

#     num_frames=241,
# #you can speficy shot_cut_frames if you wnat to, but remember to make sure shot cut number align with the text prompt.

# )

