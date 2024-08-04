import streamlit as st
from PIL import Image
import torch
import model_loader
import pipeline
from PIL import Image
# from pathlib import Path
# import matplotlib_inline as plt
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython.display import clear_output
# from transformers import CLIPTokenizer
from transformers import CLIPTokenizer
import torch


DEVICE = "cpu"

# DEVICE = torch.device("cuda")

ALLOW_CUDA = True 
ALLOW_MPS = False

if torch.cuda.is_available():
    
    DEVICE = torch.device("cuda")
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
# model_file = "../data/v1-5-pruned-emaonly.ckpt"
# models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

@st.cache_data
def load_models():
    tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
    model_file = "../data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
    return tokenizer, models

# Load models and tokenizer
tokenizer, models = load_models()

# import streamlit as st

# HTML and CSS for custom title
html_title = """
<style>
    .title-container {
        text-align: center;
        margin-top: 10px;
    }
    .title-main {
        font-size: 3em;
        font-weight: bold;
        background: -webkit-linear-gradient(45deg,violet,indigo,blue,green,yellow,orange,red);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-sub {
        font-size: 1em;
        color: black;
    }
</style>
<div class="title-container">
    <div class="title-main">Image Generator</div>
    <div class="title-sub">from Noise in Local PC</div>
</div>
"""

# Render the HTML title
st.markdown(html_title, unsafe_allow_html=True)

# You can add your code for image generation here


#
# st.title('Image Generator')
# st.title('From Pure Noise in Local PC')
# Ini
# Streamlit input fields
prompt = st.text_input("Prompt")
uncond_prompt = st.text_input("Unconditional Prompt")
# input_image = st.file_uploader("Upload an Input Image", type=["png", "jpg", "jpeg"])
strength = st.slider("Strength", 0.0, 1.0, 0.8)
cfg_scale = st.slider("CFG Scale", 0.0, 14.0, 7.5)
n_inference_steps = st.slider("Number of Inference Steps", 1, 100, 50)
seed = st.number_input("Seed", value=42)
display_steps = st.checkbox("Display Steps", value=True)
sampler = "ddpm"
image_placeholder = st.empty()
if st.button("Generate"):
    
    
    # Generate images
    final_image, images_array = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=None,
        strength=strength,
        do_cfg=True,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=n_inference_steps,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        seed=seed,
        tokenizer=tokenizer,
        display_steps=display_steps,
        stream=True,
        image_placeholder = image_placeholder
    )




    
    image_placeholder.image(final_image, caption="Final Generated Image", use_column_width=False, width=400)
