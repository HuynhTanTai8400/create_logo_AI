from flask import Flask, render_template, request
from create_logo import create_pipeline, text_2_img, model_list
import torch
from  diffusers import StableDiffusionPipeline


#Khoi tao Flask app
app = Flask(__name__)

# Dinh nghia tham so
IMAGE_PATH = "static/output.jpg"

# Khoi tao Pipeline
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

pipeline = create_pipeline()
@app.route("/", method = ["GET","POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    else:
        user_input = request.form.get["promt"]
        print("Start gen.....")
        img = text_2_img(user_input, pipeline)
        print('finish gen...')
        img.save(IMAGE_PATH)

        return render_template('index.html', image_url = IMAGE_PATH)


if __name__ == '__main__':
    app.run(debug=True)
