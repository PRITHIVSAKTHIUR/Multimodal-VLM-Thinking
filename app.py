import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread
from typing import Iterable

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
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# --- Theme and CSS Definition ---

# Define the new SteelBlue color palette
colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",  # SteelBlue base color
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue, # Use the new color
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

# Instantiate the new theme
steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""

MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

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

MODEL_ID_X = "Senqiao/VisionThink-Efficient"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True, use_fast=False)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_T = "scb10x/typhoon-ocr-3b"
processor_t = AutoProcessor.from_pretrained(MODEL_ID_T, trust_remote_code=True, use_fast=False)
model_t = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_T,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_O = "allenai/olmOCR-7B-0225-preview"
processor_o = AutoProcessor.from_pretrained(MODEL_ID_O, trust_remote_code=True, use_fast=False)
model_o = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_O,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_J = "prithivMLmods/Lumian-VLR-7B-Thinking"
SUBFOLDER = "think-preview"
processor_j = AutoProcessor.from_pretrained(MODEL_ID_J, trust_remote_code=True, subfolder=SUBFOLDER, use_fast=False)
model_j = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_J,
    trust_remote_code=True,
    subfolder=SUBFOLDER,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_V4 = 'openbmb/MiniCPM-V-4'
model_v4 = AutoModel.from_pretrained(
    MODEL_ID_V4,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).eval().to(device)
tokenizer_v4 = AutoTokenizer.from_pretrained(MODEL_ID_V4, trust_remote_code=True, use_fast=False)

MODELS = {
    "VisionThink-Efficient": (processor_x, model_x),
    "Typhoon-OCR-3B": (processor_t, model_t),
    "olmOCR-7B-0225-preview": (processor_o, model_o),
    "Lumian-VLR-7B-Thinking": (processor_j, model_j),
}

def downsample_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
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
    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

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

    if model_name not in MODELS:
        yield "Invalid model selected.", "Invalid model selected."
        return
    processor, model = MODELS[model_name]

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full], images=[image], return_tensors="pt", padding=True).to(device)
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
    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames_with_ts = downsample_video(video_path)
    if not frames_with_ts:
        yield "Could not process video.", "Could not process video."
        return

    if model_name == "openbmb/MiniCPM-V-4":
        images = [frame for frame, ts in frames_with_ts]
        content = [text] + images
        msgs = [{'role': 'user', 'content': content}]
        try:
            answer = model_v4.chat(
                image=images[0].convert('RGB'), msgs=msgs, tokenizer=tokenizer_v4,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, repetition_penalty=repetition_penalty,
            )
            yield answer, answer
        except Exception as e:
            yield f"Error: {e}", f"Error: {e}"
        return

    if model_name not in MODELS:
        yield "Invalid model selected.", "Invalid model selected."
        return
    processor, model = MODELS[model_name]

    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    images_for_processor = []
    for frame, timestamp in frames_with_ts:
        messages[0]["content"].insert(0, {"type": "image"})
        images_for_processor.append(frame)

    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full], images=images_for_processor, return_tensors="pt", padding=True).to(device)
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

with gr.Blocks(theme=steel_blue_theme, css=css) as demo:
    gr.Markdown("# **Multimodal VLM Thinking**", elem_id="main-title")
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image", height=290)
                    image_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=image_examples, inputs=[image_query, image_upload])
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Video", height=290)
                    video_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=video_examples, inputs=[video_query, video_upload])

            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)

        with gr.Column(scale=3):
            gr.Markdown("## Output", elem_id="output-title")
            output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=11, show_copy_button=True)
            with gr.Accordion("(Result.md)", open=False):
                markdown_output = gr.Markdown(label="(Result.Md)")

            model_choice = gr.Radio(
                choices=["Lumian-VLR-7B-Thinking", "VisionThink-Efficient", "openbmb/MiniCPM-V-4", "Typhoon-OCR-3B", "olmOCR-7B-0225-preview"],
                label="Select Model",
                value="Lumian-VLR-7B-Thinking"
            )
            
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
    demo.queue(max_size=50).launch(mcp_server=True, ssr_mode=False, show_error=True)
