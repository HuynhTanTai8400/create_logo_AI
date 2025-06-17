from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image

# Mô hình và prompt
# Sửa model_list thành biến đơn hoặc danh sách nếu có nhiều model
model_name_sd = "runwayml/stable-diffusion-v1-5"


# Dinh nghia tham so
rand_seed = torch.manual_seed(0)
NUM_INFERENCE_STEPS = 20 # Sửa thành NUM_INFERENCE_STEPS (có 'S')
GUIDANCE_SCALE = 7.5 # Điều chỉnh GUIDANCE_SCALE lên giá trị hợp lý hơn
HEIGHT = 512
WIDTH = 512


def create_pipeline(model_name=model_name_sd): # Sửa tham số mặc định
    # Neu may co GPU cuda (se nhanh hon)
    if torch.cuda.is_available():
        print('Using GPU')
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype = torch.float16, # Nên dùng float16 cho GPU để tối ưu
            use_safetensors = True
        ).to("cuda")
    else:
        print("Using CPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype = torch.float32, # Giữ float32 cho CPU
            use_safetensors = True
        )
    # Có thể thêm scheduler nếu muốn cải thiện chất lượng/tốc độ (ví dụ DPMSolverMultistepScheduler)
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return pipeline

def text_2_img(prompt, pipeline): 
    images = pipeline(
        prompt, 
        guidance_scale = GUIDANCE_SCALE,
        num_inference_steps = NUM_INFERENCE_STEPS, 
        generator = rand_seed,
        # num_images_per_request = 1, 
        height = HEIGHT,
        width = WIDTH,
    ).images
    return images[0]