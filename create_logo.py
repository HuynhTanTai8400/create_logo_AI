from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image

# Mô hình và prompt
model_list = "runwayml/stable-diffusion-v1-5"
prompt = "logo, street food stall, 'Quán Ngon', **cartoon, illustration, flat design**. Cute smiling food cart character, holding a bowl of noodles. **Vibrant, warm colors, red, yellow, orange accents**. Solid light blue background. Simple, eye-catching. --ar 1:1 --v 5.2"

# Dinh nghia tham so
rand_seed = torch.manual_seed(0)
NUM_INFERENCE_STEP = 20
GUIDANCE_SCALE = 0.75
HEIGHT = 512
WIDTH = 512



def create_pipeline(model_name= model_list[0]):
    # Neu may co GPU cuda (se nhanh hon)
    if torch.cuda.is_available():
        print('using GPU')
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype = torch.float32,
            use_safetensors = True
        ).to("cuda")
    else:
        print("Using CPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype = torch.float32,
            use_safetensors = True
             )
    return pipeline

def text_2_img(promt, pipeline):
    images = pipeline(
        promt,
        guidance_scale = GUIDANCE_SCALE,
        num_inference_step = NUM_INFERENCE_STEP,
        generator = rand_seed,
        num_images_per_request =1,
        height = HEIGHT,
        width = WIDTH,
    ).images
    return images[0]








