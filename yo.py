from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
import numpy as np
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(img_input):
    if img_input is None:
        return "Please upload an image or select an example."
        
    if isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input)
    else:
        img = img_input

    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

example_dir = "examples"
example_paths = []
if os.path.exists(example_dir):
    example_paths = [os.path.join(example_dir, img) for img in os.listdir(example_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

custom_css = """
h1 {
    background: -webkit-linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em !important;
}
#output-box {
    border: 2px solid #4facfe;
    border-radius: 10px;
    background-color: #f9f9f9;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Image Caption Generator", css=custom_css) as app:
    gr.Markdown(
        """
        # 📸 Image Caption Generator
        """
    )
    with gr.Row():
        
        with gr.Column():
            caption_output = gr.Textbox(label="Generated Caption", interactive=False, elem_id="output-box")

        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Your Image Here")
            submit_button = gr.Button("Generate Caption", variant="primary")
            
            gr.Examples(
                examples=example_paths,
                inputs=image_input,
                outputs=caption_output,
                fn=generate_caption,
                cache_examples=True
            )

    submit_button.click(
        fn=generate_caption,
        inputs=image_input,
        outputs=caption_output
    )

app.launch(share=True)