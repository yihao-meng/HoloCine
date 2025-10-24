# HoloCine: Holistic Generation of Cinematic Multi-Shot Long Video Narratives


**[**[**📄 Paper**](https://arxiv.org/abs/2510.20822)**]**
**[**[**🌐 Project Page**](https://holo-cine.github.io/)**]**
**[**[**🤗 Model Weights**](https://huggingface.co/hlwang06/HoloCine/tree/main)**]**



https://github.com/user-attachments/assets/c4dee993-7c6c-4604-a93d-a8eb09cfd69b



_**[Yihao Meng<sup>1,2</sup>](https://yihao-meng.github.io/), [Hao Ouyang<sup>2</sup>](https://ken-ouyang.github.io/), [Yue Yu<sup>1,2</sup>](https://bruceyy.com/), [Qiuyu Wang<sup>2</sup>](https://github.com/qiuyu96), [Wen Wang<sup>2,3</sup>](https://github.com/encounter1997), [Ka Leong Cheng<sup>2</sup>](https://felixcheng97.github.io/), <br>[Hanlin Wang<sup>1,2</sup>](https://scholar.google.com/citations?user=0uO4fzkAAAAJ&hl=zh-CN), [Yixuan Li<sup>2,4</sup>](https://yixuanli98.github.io/), [Cheng Chen<sup>2,5</sup>](https://scholar.google.com/citations?user=nNQU71kAAAAJ&hl=zh-CN), [Yanhong Zeng<sup>2</sup>](https://zengyh1900.github.io/), [Yujun Shen<sup>2</sup>](https://shenyujun.github.io/), [Huamin Qu<sup>1</sup>](http://huamin.org/)**_
<br>
<sup>1</sup>HKUST, <sup>2</sup>Ant Group, <sup>3</sup>ZJU, <sup>4</sup>CUHK, <sup>5</sup>NTU

# TLDR
*   **What it is:** A text-to-video model that generates full scenes, not just isolated clips.
*   **Key Feature:** It maintains consistency of characters, objects, and style across all shots in a scene.
*   **How it works:** You provide shot-by-shot text prompts, giving you directorial control over the final video.

**Strongly recommend seeing our [demo page](https://holo-cine.github.io/).**

If you enjoyed the videos we created, please consider giving us a star 🌟.

## 🚀 Open-Source Plan

### ✅ Released
*   Full inference code
*   `HoloCine-14B-full` 
*   `HoloCine-14B-sparse`

### ⏰ To Be Released
*   `HoloCine-14B-full-l` (For videos longer than 1 minute)
*   `HoloCine-14B-sparse-l` (For videos longer than 1 minute)
*   `HoloCine-5B-full` (For limited-memory users)
*   `HoloCine-5B-sparse` (For limited-memory users)

### 🗺️ In Planning
*   `HoloCine-audio`

# Setup
```shell
git clone https://github.com/yihao-meng/HoloCine.git
cd HoloCine
```
# Environment
We use a environment similar to diffsynth. If you have a diffsynth environment, you can probably reuse it.
```shell
conda create -n HoloCine python=3.10
pip install -e .
```

We use FlashAttention-3 to implement the sparse inter-shot attention. We highly recommend using FlashAttention-3 for its fast speed. We provide a simple instruction on how to install FlashAttention-3.

```shell
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
cd hopper
python setup.py install
```
If you encounter environment problem when installing FlashAttention-3, you can refer to their official github page https://github.com/Dao-AILab/flash-attention.

If you cannot install FlashAttention-3, you can use FlashAttention-2 as an alternative, and our code will automatically detect the FlashAttention version. It will be slower than FlashAttention-3,but can also produce the right result.

If you want to install FlashAttention-2, you can use the following command:
```shell
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

# Checkpoint


### Step 1: Download Wan 2.2 VAE and T5
If you already have downloaded Wan 2.2 14B T2V before, skip this section.

If not, you need the T5 text encoder and the VAE from the original Wan 2.2 repository:
[https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)


Based on the repository's file structure, you **only** need to download `models_t5_umt5-xxl-enc-bf16.pth` and `Wan2.1_VAE.pth`.

You do **not** need to download the `google`, `high_noise_model`, or `low_noise_model` folders, nor any other files. 

#### Recommended Download (CLI)

We recommend using `huggingface-cli` to download only the necessary files. Make sure you have `huggingface_hub` installed (`pip install huggingface_hub`).

This command will download *only* the required T5 and VAE models into the correct directory:

```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B \
  --local-dir checkpoints/Wan2.2-T2V-A14B \
  --allow-patterns "models_t5_*.pth" "Wan2.1_VAE.pth"
```

#### Manual Download

Alternatively, go to the "Files" tab on the Hugging Face repo and manually download the following two files:

  * `models_t5_umt5-xxl-enc-bf16.pth`
  * `Wan2.1_VAE.pth`

Place both files inside a new folder named `checkpoints/Wan2.2-T2V-A14B/`.

### Step 2: Download HoloCine Model (HoloCine\_dit)

Download our fine-tuned high-noise and low-noise DiT checkpoints from the following link:

**[➡️ Download HoloCine\_dit Model Checkpoints [Here](https://huggingface.co/hlwang06/HoloCine)]**

This download contain the four fine-tuned model files. Two for full_attention version: `full_high_noise.safetensors`, `full_low_noise.safetensors`. And two for sparse inter-shot attention version: `sparse_high_noise.safetensors`, `sparse_high_noise.safetensors`. The sparse version is still uploading. 

You can choose a version to download, or try both version if you want. 

The full attention version will have better performance, so we suggest you start from it. The sparse inter-shot attention version will be slightly unstable (but also great in most cases), but faster than the full attention version.

For full attention version:
Create a new folder named `checkpoints/HoloCine_dit/full/` and place both high and low noise files inside.

For sparse attention version:
Create a new folder named `checkpoints/HoloCine_dit/full/` and place both high and low noise files inside.
### Step 3: Final Directory Structure

If you downloaded the `full` model, your `checkpoints` directory should look like this:

```
checkpoints/
├── Wan2.2-T2V-A14B/
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── Wan2.1_VAE.pth
└── HoloCine_dit/
    └── full/
        ├── full_high_noise.safetensors
        └── full_low_noise.safetensors
```
(If you downloaded the `sparse` model, replace `full` with `sparse`.)


# Inference
We release two version of models, one using full attention to model the multi-shot sequence (our default), the other using sparse Inter-shot attention. 

To use the full attention version.

```shell
python HoloCine_inference_full_attention.py
```

To use the sparse inter-shot attention version.

```shell
python HoloCine_inference_sparse_attention.py
```




## Prompt Format

To achieve precise content control of each shot, our prompt is designed to follow a format. Our inference script is designed to be flexible and we support two way to input the text prompt.

### Choice 1: Structured Input (Recommended if you want to test on your own sample)

This is the easiest way to create new multi-shot prompts. You provide the components as separate arguments inside the script, and our helper function will format them correctly.

  * `global_caption`: A string describing the entire scene, characters, and setting.
  * `shot_captions`: A *list* of strings, where each string describes one shot in sequential order.
  * `num_frames`: The total number of frames for the video (default is `241` as we train on this sequence length).
  * `shot_cut_frames`: (Optional) A list of frame numbers where you want cuts to happen. By defult, the script will automatically calculate evenly spaced cuts. If you want to customize it, make sure you understand that the shot cut number indicated by `shot_cut_frames` should align with `shot_captions`.

**Example (inside `HoloCine_inference_full_attention.py`):**

```python
run_inference(
    pipe=pipe,
    negative_prompt=scene_negative_prompt,
    output_path="test_structured_output.mp4",
    
    # Choice 1 inputs
    global_caption="The scene is set in a lavish, 1920s Art Deco ballroom during a masquerade party. [character1] is a mysterious woman with a sleek bob, wearing a sequined silver dress and an ornate feather mask. [character2] is a dapper gentleman in a black tuxedo, his face half-hidden by a simple black domino mask. The environment is filled with champagne fountains, a live jazz band, and dancing couples in extravagant costumes. This scene contains 5 shots.",
    shot_captions=[
        "Medium shot of [character1] standing by a pillar, observing the crowd, a champagne flute in her hand.",
        "Close-up of [character2] watching her from across the room, a look of intrigue on his visible features.",
        "Medium shot as [character2] navigates the crowd and approaches [character1], offering a polite bow. ",
        "Close-up on [character1]'s eyes through her mask, as they crinkle in a subtle, amused smile.",
        "A stylish medium two-shot of them standing together, the swirling party out of focus behind them, as they begin to converse."

    ],
    num_frames=241
)
```

https://github.com/user-attachments/assets/10dba757-27dc-4f65-8fc3-b396cf466063

### Choice 2: Raw String Input

This mode allows you to provide the full, concatenated prompt string, just like in our original script. This is useful if you want to re-using our provided prompts.

The format must be exact:
`[global caption] ... [per shot caption] ... [shot cut] ... [shot cut] ...`

**Example (inside `HoloCine_inference_full_attention.py`):**

```python
run_inference(
    pipe=pipe,
    negative_prompt=scene_negative_prompt,
    output_path="test_raw_string_output.mp4",
    
    # Choice 2 inputs
    prompt="[global caption] The scene features a young painter, [character1], with paint-smudged cheeks and intense, focused eyes. Her hair is tied up messily. The setting is a bright, sun-drenched art studio with large windows, canvases, and the smell of oil paint. This scene contains 6 shots. [per shot caption] Medium shot of [character1] standing back from a large canvas, brush in hand, critically observing her work. [shot cut] Close-up of her hand holding the brush, dabbing it thoughtfully onto a palette of vibrant colors. [shot cut] Extreme close-up of her eyes, narrowed in concentration as she studies the canvas. [shot cut] Close-up on the canvas, showing a detailed, textured brushstroke being slowly applied. [shot cut] Medium close-up of [character1]'s face, a small, satisfied smile appears as she finds the right color. [shot cut] Over-the-shoulder shot showing her add a final, delicate highlight to the painting.",
    

    num_frames=241,
    shot_cut_frames=[37, 73, 113, 169, 205]

)
```
https://github.com/user-attachments/assets/fdc12ff1-cf1b-4250-b7c9-a32e4d65731f

## Examples

We provide several commented-out examples directly within the `HoloCine_inference_full_attention.py` and `HoloCine_inference_sparse_attention.py` script. You can uncomment any of these examples to try them out immediately.

If you want to quickly test the model's stability on your own text prompt and don't want to design it by yourself, you can use LLM like gemini 2.5 pro to generate text prompt based on our format. Based on our test, the model is quite stable on diverse genres of text prompt.



# Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{meng2025holocine,
  title={HoloCine: Holistic Generation of Cinematic Multi-Shot Long Video Narratives},
  author={Meng, Yihao and Ouyang, Hao and Yu, Yue and Wang, Qiuyu and Wang, Wen and Cheng, Ka Leong and Wang, Hanlin and Li, Yixuan and Chen, Cheng and Zeng, Yanhong and Shen, Yujun and Qu, Huamin},
  journal={arXiv preprint arXiv:2510.20822},
  year={2025}
}
```

# License

This project is licensed under the CC BY-NC-SA 4.0 ([Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

The code is provided for academic research purposes only.

For any questions, please contact ymengas@cse.ust.hk.

