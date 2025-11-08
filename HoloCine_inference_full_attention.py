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
    """
    (Helper for Mode 1)
    Prepares the inference parameters from user-friendly segmented inputs.
    """
    
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
    
    Mode 1 (Structured): Provide 'global_caption', 'shot_captions', 'num_frames'.
                         'shot_cut_frames' is optional (auto-calculated).
    Mode 2 (Raw): Provide 'prompt'.
                  'num_frames' and 'shot_cut_frames' are optional.
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
        ModelConfig(path="./checkpoints/Wan2.2-T2V-A14B//Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(path="./checkpoints/HoloCine_dit/full/full_high_noise.safetensors", offload_device="cpu"),
        ModelConfig(path="./checkpoints/HoloCine_dit/full/full_low_noise.safetensors",  offload_device="cpu"),
        ModelConfig(path="./checkpoints/Wan2.2-T2V-A14B/Wan2.2-T2V-A14B/Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()
pipe.to(device)

# --- 2. Define Common Parameters ---
scene_negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


# ===================================================================
#                ✨ How to Use ✨
# ===================================================================

# --- Example 1: Call using Structured Input (Choice 1) ---
# (Auto-calculates shot cuts)
print("\n--- Running Example 1 (Structured Input) ---")
run_inference(
    pipe=pipe,
    negative_prompt=scene_negative_prompt,
    output_path="video1.mp4",
    
    # Choice 1 inputs
    global_caption="The scene is set in a lavish, 1920s Art Deco ballroom during a masquerade party. [character1] is a mysterious woman with a sleek bob, wearing a sequined silver dress and an ornate feather mask. [character2] is a dapper gentleman in a black tuxedo, his face half-hidden by a simple black domino mask. The environment is filled with champagne fountains, a live jazz band, and dancing couples in extravagant costumes. This scene contains 5 shots.",
    shot_captions=[
        "Medium shot of [character1] standing by a pillar, observing the crowd, a champagne flute in her hand.",
        "Close-up of [character2] watching her from across the room, a look of intrigue on his visible features.",
        "Medium shot as [character2] navigates the crowd and approaches [character1], offering a polite bow.",
        "Close-up on [character1]'s eyes through her mask, as they crinkle in a subtle, amused smile.",
        "A stylish medium two-shot of them standing together, the swirling party out of focus behind them, as they begin to converse."

    ],
    num_frames=241
)


# --- Example 2: Call using Raw String Input (Choice 2) ---
# (Uses your original prompt format)
print("\n--- Running Example 2 (Raw String Input) ---")
run_inference(
    pipe=pipe,
    negative_prompt=scene_negative_prompt,
    output_path="video2.mp4",
    
    # Choice 2 inputs
    prompt="[global caption] The scene features a young painter, [character1], with paint-smudged cheeks and intense, focused eyes. Her hair is tied up messily. The setting is a bright, sun-drenched art studio with large windows, canvases, and the smell of oil paint. This scene contains 6 shots. [per shot caption] Medium shot of [character1] standing back from a large canvas, brush in hand, critically observing her work. [shot cut] Close-up of her hand holding the brush, dabbing it thoughtfully onto a palette of vibrant colors. [shot cut] Extreme close-up of her eyes, narrowed in concentration as she studies the canvas. [shot cut] Close-up on the canvas, showing a detailed, textured brushstroke being slowly applied. [shot cut] Medium close-up of [character1]'s face, a small, satisfied smile appears as she finds the right color. [shot cut] Over-the-shoulder shot showing her add a final, delicate highlight to the painting.",
    num_frames=241,  
    shot_cut_frames=[37, 73, 113, 169, 205]
)


# # we provide more samples for test, you can uncomment them and have a try.
# run_inference(
#     pipe=pipe,
#     negative_prompt=scene_negative_prompt,
#     output_path="video3.mp4",
    
#     # Choice 2 inputs
#     prompt="[global caption] The scene is set in a gritty, underground boxing gym. [character1] is a weary, aging boxing coach with a towel around his neck and a kind but tough face. [character2] is a young, ambitious female boxer with a focused, intense expression, her hands wrapped. The environment smells of sweat and leather, with punching bags, a boxing ring, and faded posters on the brick walls. This scene contains 6 shots. [per shot caption] Medium shot of [character2] relentlessly hitting a heavy bag. [shot cut] Close-up of [character1] watching her, his expression critical yet proud. [shot cut] Close-up of [character2]'s determined face, dripping with sweat, her eyes fixed on the bag. [shot cut] Medium shot as [character1] approaches and stops the bag, giving her a piece of advice. [shot cut] Close-up of [character2] listening intently, nodding in understanding. [shot cut] Medium shot as she turns back to the bag and starts punching again, her form now visibly improved.",
#     num_frames=241,  
#     shot_cut_frames=[37, 77, 117, 157, 197]
# )




# run_inference(
#     pipe=pipe,
#     negative_prompt=scene_negative_prompt,
#     output_path="video4.mp4",
    
#     # Choice 2 inputs
#     prompt="[global caption] The scene is a magical encounter in a misty, ancient Celtic ruin at dawn. [character1] is a modern-day historian, a skeptical woman with practical hiking gear and a camera. [character2] is the spectral figure of an ancient Celtic queen, translucent and ethereal, with long, flowing red hair and a silver circlet. The environment is comprised of mossy standing stones and rolling green hills shrouded in morning mist. This scene contains 5 shots. [per shot caption] Medium shot of [character1] carefully touching a moss-covered standing stone, a look of academic interest on her face. [shot cut] Close-up of her face, her expression changing to one of utter shock as she sees something off-camera. [shot cut] A soft-focus shot of [character2] slowly materializing from the mist between two stones. [shot cut] Medium shot of [character1] stumbling backward, lowering her camera, her skepticism completely shattered. [shot cut] Close-up of [character2]'s spectral face, her expression sad and timeless as she looks at the historian.",
#     num_frames=241,  
#     shot_cut_frames=[49, 93, 137, 189],
# )


# run_inference(
#     pipe=pipe,
#     negative_prompt=scene_negative_prompt,
#     output_path="video5.mp4",
    
#     # Choice 2 inputs
#     prompt="[global caption] The scene is set in an enchanted, bioluminescent forest at twilight. [character1] is an ancient elf with long, silver hair braided with glowing flowers, wearing ethereal white robes. [character2] is a lost human child with short, messy brown hair and wide, fearful eyes, clutching a wooden toy. The environment is filled with giant, glowing mushrooms, sparkling flora, and shafts of moonlight breaking through a thick canopy. This scene contains 5 shots. [per shot caption] Medium shot of [character2] hiding behind a large, glowing mushroom, peering out nervously. [shot cut] Close-up of [character1]'s hand, fingers adorned with delicate rings, gently touching a luminous plant, causing it to glow brighter. [shot cut] Medium shot of [character1] turning their head, their pointed ears catching the faint sound of the child's whimper. [shot cut] Close-up of [character2]'s face, a tear rolling down their cheek, illuminated by the blue light of the forest. [shot cut] A soft-focus shot from the child's perspective, showing [character1] approaching slowly with a kind, reassuring smile, their form haloed by the forest's light.",
#     num_frames=241,  
#     shot_cut_frames=[49, 93, 137, 189],
# )


