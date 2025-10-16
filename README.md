# **Multimodal VLM Thinking**

<img width="1794" height="1120" alt="Screenshot 2025-10-16 at 11-56-58 Multimodal VLM Thinking - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/3bf78eb4-3cf7-4996-a019-32009a9a7e2a" />

A comprehensive Gradio application that provides access to multiple state-of-the-art Vision-Language Models (VLMs) for both image and video understanding tasks. This application offers a unified interface to interact with various specialized models for OCR, document analysis, visual reasoning, and multimodal understanding.

## Features

- **Multi-Model Support**: Access to 5 different VLMs in a single interface
- **Image & Video Processing**: Support for both static image and dynamic video analysis
- **Real-time Streaming**: Live text generation with streaming responses
- **Advanced Configuration**: Customizable generation parameters
- **User-friendly Interface**: Clean Gradio web interface with examples

## Supported Models

### Primary Models

1. **Lumian-VLR-7B-Thinking** (Default)
   - 7B parameter vision-language reasoning model
   - Built on Qwen2.5-VL-7B-Instruct
   - Specialized in fine-grained multimodal understanding and video reasoning
   - Explicit grounded reasoning capabilities

2. **MiniCPM-V-4**
   - 4.1B parameters (SigLIP2-400M + MiniCPM4-3B)
   - Latest efficient model in the MiniCPM-V series
   - Strong single-image, multi-image, and video understanding
   - Optimized for efficiency

3. **Typhoon-OCR-3B**
   - 3B parameter OCR-specialized model
   - Optimized for optical character recognition
   - Efficient performance in challenging conditions

4. **DREX-062225-7B-exp** (Experimental)
   - Experimental multimodal model
   - Strong document reading and extraction capabilities
   - Advanced vision-language understanding

5. **olmOCR-7B-0225-preview**
   - 7B parameter OCR model by AllenAI
   - Robust text extraction from complex document layouts
   - Preview version with advanced capabilities

## Installation

### Prerequisites

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install gradio
pip install spaces
pip install opencv-python
pip install pillow
pip install numpy
pip install requests
```

### Clone Repository

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Multimodal-VLM-Thinking.git
cd Multimodal-VLM-Thinking
```

### Run Application

```bash
python app.py
```

## Usage

### Image Inference

1. Navigate to the "Image Inference" tab
2. Enter your query in the text box
3. Upload an image file
4. Select your preferred model
5. Adjust advanced parameters if needed
6. Click "Submit"

### Video Inference

1. Navigate to the "Video Inference" tab
2. Enter your query describing what you want to analyze
3. Upload a video file
4. Select your preferred model
5. Configure generation parameters
6. Click "Submit"

### Example Use Cases

- **Document Analysis**: "Convert this page to doc [markdown] precisely."
- **Safety Assessment**: "Describe the safety measures in the image. Conclude (Safe / Unsafe)."
- **Creative Analysis**: "Explain the creativity in the image."
- **Chart Conversion**: "Convert chart to OTSL."
- **Video Understanding**: "Explain the video in detail."

## Configuration Parameters

### Generation Settings

- **Max New Tokens**: 1-4096 (default: 2048)
- **Temperature**: 0.1-4.0 (default: 0.6)
- **Top-p**: 0.05-1.0 (default: 0.9)
- **Top-k**: 1-1000 (default: 50)
- **Repetition Penalty**: 1.0-2.0 (default: 1.2)

## Technical Details

### Video Processing

- Automatically downsamples videos to 10 evenly spaced frames
- Maintains aspect ratio and quality
- Includes timestamp information for temporal understanding

### Model Loading

- All models loaded with float16 precision for efficiency
- Automatic device detection (CUDA/CPU)
- Trust remote code enabled for specialized model architectures

### Memory Management

- Uses Hugging Face Spaces GPU decorator for efficient resource allocation
- Streaming text generation to reduce memory footprint
- Optimized batch processing for multiple inputs

## Hardware Requirements

### Recommended

- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space for model weights

### Minimum

- **GPU**: NVIDIA GPU with 65GB+ VRAM
- **RAM**: 32GB+ system memory
- **Storage**: 70GB+ free space

## API Integration

The application can be easily integrated into other systems:

```python
# Example integration
from your_app import generate_image, generate_video

# Image processing
result = generate_image(
    model_name="Lumian-VLR-7B-Thinking",
    text="Describe this image",
    image=your_pil_image,
    max_new_tokens=1024
)

# Video processing
result = generate_video(
    model_name="MiniCPM-V-4",
    text="Analyze this video",
    video_path="path/to/video.mp4",
    max_new_tokens=2048
)
```

## Model Performance Comparison

| Model | Parameters | Strengths | Best For |
|-------|------------|-----------|----------|
| Lumian-VLR-7B | 7B | Reasoning, Video | Complex analysis |
| MiniCPM-V-4 | 4.1B | Efficiency, Multi-image | General purpose |
| Typhoon-OCR-3B | 3B | OCR, Speed | Text extraction |
| olmOCR-7B | 7B | Document layouts | Complex documents |
| DREX-062225 | 7B | Experimental features | Research tasks |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## Acknowledgments

- Hugging Face for model hosting and transformers library
- Gradio for the web interface framework
- Model creators: Qwen team, MiniCPM team, SCB 10X, AllenAI

## Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check Hugging Face model cards for detailed model information

## Updates

Check the GitHub repository for the latest updates and model additions. The application is actively maintained with regular improvements and new model integrations.
