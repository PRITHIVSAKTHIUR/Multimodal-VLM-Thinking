import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2
import requests

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
    AutoModel,
    AutoTokenizer,
)
from transformers.image_utils import load_image

# Constants for text generation
MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 2048
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# Let the environment (e.g., Hugging Face Spaces) determine the device.
# This avoids conflicts with the CUDA environment setup by the platform.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)
# --- Model Loading ---

# To address the warnings, we add `use_fast=False` to ensure we use the
# processor version the model was originally saved with.

# Load DREX-062225-exp
MODEL_ID_X = "prithivMLmods/DREX-062225-exp"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True, use_fast=False)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load typhoon-ocr-3b
MODEL_ID_T = "scb10x/typhoon-ocr-3b"
processor_t = AutoProcessor.from_pretrained(MODEL_ID_T, trust_remote_code=True, use_fast=False)
model_t = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_T,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load olmOCR-7B-0225-preview
MODEL_ID_O = "allenai/olmOCR-7B-0225-preview"
processor_o = AutoProcessor.from_pretrained(MODEL_ID_O, trust_remote_code=True, use_fast=False)
model_o = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_O,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Lumian-VLR-7B-Thinking
MODEL_ID_J = "prithivMLmods/Lumian-VLR-7B-Thinking"
SUBFOLDER = "think-preview"
processor_j = AutoProcessor.from_pretrained(MODEL_ID_J, trust_remote_code=True, subfolder=SUBFOLDER, use_fast=False)
model_j = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_J,
    trust_remote_code=True,
    subfolder=SUBFOLDER,
    torch_dtype=torch.float16
).to(device).eval()

# Load openbmb/MiniCPM-V-4
MODEL_ID_V4 = 'openbmb/MiniCPM-V-4'
model_v4 = AutoModel.from_pretrained(
    MODEL_ID_V4,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    # Using 'sdpa' can sometimes cause issues in certain environments,
    # letting transformers choose the default is safer.
    # attn_implementation='sdpa'
).eval().to(device)
tokenizer_v4 = AutoTokenizer.from_pretrained(MODEL_ID_V4, trust_remote_code=True, use_fast=False)

# --- Refactored Model Dictionary ---
# This simplifies model selection in the generation functions.
MODELS = {
    "DREX-062225-7B-exp": (processor_x, model_x),
    "Typhoon-OCR-3B": (processor_t, model_t),
    "olmOCR-7B-0225-preview": (processor_o, model_o),
    "Lumian-VLR-7B-Thinking": (processor_j, model_j),
}


def downsample_video(video_path):
    """
    Downsamples the video to evenly spaced frames.
    Each frame is returned as a PIL image along with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    # Use a maximum of 10 frames to avoid excessive memory usage
    frame_indices = np.linspace(0, total_frames - 1, min(total_frames, 10), dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for image input.
    """
    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    # Handle MiniCPM-V-4 separately due to its different API
    if model_name == "openbmb/MiniCPM-V-4":
        msgs = [{'role': 'user', 'content': [image, text]}]
        try:
            answer = model_v4.chat(
                image=image.convert('RGB'), msgs=msgs, tokenizer=tokenizer_v4,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, repetition_penalty=repetition_penalty,
            )
            yield answer, answer
        except Exception as e:
            yield f"Error: {e}", f"Error: {e}"
        return

    # Use the dictionary for other models
    if model_name not in MODELS:
        yield "Invalid model selected.", "Invalid model selected."
        return
    processor, model = MODELS[model_name]

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full], images=[image], return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_video(model_name: str, text: str, video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for video input.
    """
    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames_with_ts = downsample_video(video_path)
    if not frames_with_ts:
        yield "Could not process video.", "Could not process video."
        return

    # Handle MiniCPM-V-4 separately
    if model_name == "openbmb/MiniCPM-V-4":
        images = [frame for frame, ts in frames_with_ts]
        # For video, the prompt includes the text and then all the image frames
        content = [text] + images
        msgs = [{'role': 'user', 'content': content}]
        try:
            # The .chat API still takes a single image argument, typically the first frame
            answer = model_v4.chat(
                image=images[0].convert('RGB'), msgs=msgs, tokenizer=tokenizer_v4,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, repetition_penalty=repetition_penalty,
            )
            yield answer, answer
        except Exception as e:
            yield f"Error: {e}", f"Error: {e}"
        return

    # Use the dictionary for other models
    if model_name not in MODELS:
        yield "Invalid model selected.", "Invalid model selected."
        return
    processor, model = MODELS[model_name]

    # Prepare messages for Qwen-style models
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    images_for_processor = []
    for frame, timestamp in frames_with_ts:
        messages[0]["content"].append({"type": "image", "image": frame})
        images_for_processor.append(frame)

    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full], images=images_for_processor, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs, "streamer": streamer, "max_new_tokens": max_new_tokens,
        "do_sample": True, "temperature": temperature, "top_p": top_p,
        "top_k": top_k, "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer


# Define examples for image and video inference
image_examples = [
    ["Describe the safety measures in the image. Conclude (Safe / Unsafe)..", "images/5.jpg"],
    ["Convert this page to doc [markdown] precisely.", "images/3.png"],
    ["Convert this page to doc [markdown] precisely.", "images/4.png"],
    ["Explain the creativity in the image.", "images/6.jpg"],
    ["Convert this page to doc [markdown] precisely.", "images/1.png"],
    ["Convert chart to OTSL.", "images/2.png"]
]

video_examples = [
    ["Explain the video in detail.", "videos/2.mp4"],
    ["Explain the ad in detail.", "videos/1.mp4"]
]

css = """
.submit-btn { background-color: #2980b9 !important; color: white !important; }
.submit-btn:hover { background-color: #3498db !important; }
.canvas-output { border: 2px solid #4682B4; border-radius: 10px; padding: 20px; }
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# **[Multimodal VLM Thinking](https://huggingface.co/collections/prithivMLmods/multimodal-implementations-67c9982ea04b39f0608badb0)**")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image")
                    image_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(examples=image_examples, inputs=[image_query, image_upload])
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Video")
                    video_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(examples=video_examples, inputs=[video_query, video_upload])

            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)

        with gr.Column():
            with gr.Column(elem_classes="canvas-output"):
                gr.Markdown("## Output")
                output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=3, show_copy_button=True)
                with gr.Accordion("(Result.md)", open=False):
                    markdown_output = gr.Markdown(label="(Result.Md)")
            model_choice = gr.Radio(
                choices=["Lumian-VLR-7B-Thinking", "openbmb/MiniCPM-V-4", "Typhoon-OCR-3B", "DREX-062225-7B-exp", "olmOCR-7B-0225-preview"],
                label="Select Model",
                value="Lumian-VLR-7B-Thinking"
            )
            gr.Markdown("**Model Info 💻** | [Report Bug](https://huggingface.co/spaces/prithivMLmods/Multimodal-VLM-Thinking/discussions)")
            gr.Markdown("> [MiniCPM-V 4.0](https://huggingface.co/openbmb/MiniCPM-V-4) is the latest efficient model in the MiniCPM-V series. The model is built based on SigLIP2-400M and MiniCPM4-3B with a total of 4.1B parameters. It inherits the strong single-image, multi-image and video understanding performance of MiniCPM-V 2.6 with largely improved efficiency. [Lumian-VLR-7B-Thinking](https://huggingface.co/prithivMLmods/Lumian-VLR-7B-Thinking) is a high-fidelity vision-language reasoning model built on Qwen2.5-VL-7B-Instruct, designed for fine-grained multimodal understanding, video reasoning, and document comprehension through explicit grounded reasoning.")
            gr.Markdown("> [olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview) is a 7B parameter open large model designed for OCR tasks with robust text extraction, especially in complex document layouts. [Typhoon-ocr-3b](https://huggingface.co/scb10x/typhoon-ocr-3b) is a 3B parameter OCR model optimized for efficient and accurate optical character recognition in challenging conditions.")
            gr.Markdown("> [DREX-062225-exp](https://huggingface.co/prithivMLmods/DREX-062225-exp) is an experimental multimodal model emphasizing strong document reading and extraction capabilities combined with vision-language understanding to support detailed document parsing and reasoning tasks.")
            gr.Markdown("> ⚠️ Note: Video inference performance can vary significantly between models.")

    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[output, markdown_output]
    )
    video_submit.click(
        fn=generate_video,
        inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[output, markdown_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(share=True, ssr_mode=False, show_error=True)
