from flask import Flask, render_template, request
from create_logo import create_pipeline, text_2_img, model_list
import torch
from  diffusers import StableDiffusionPipeline


#Khoi tao Flask app
app = Flask(__name__)

# Dinh nghia tham so
IMAGE_PATH = "static/output.jpg"


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

