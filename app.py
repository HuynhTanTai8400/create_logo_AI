from flask import Flask, render_template, request
from create_logo import create_pipeline, text_2_img
import torch
import os # Thêm import os

# Khoi tao Flask app
app = Flask(__name__)

# Dinh nghia tham so
IMAGE_PATH = "static/output.jpg"

pipeline = create_pipeline()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    else:
        # Sửa "promt" thành "prompt" và thêm dấu ngoặc đơn cho .get()
        user_input = request.form.get("prompt")
        if not user_input:
            # Xử lý trường hợp không có prompt
            return render_template('index.html', error_message="Vui lòng nhập prompt để tạo ảnh.")

        print("Start gen.....")
        try:
            img = text_2_img(user_input, pipeline)
            print('finish gen...')
            img.save(IMAGE_PATH)
            return render_template('index.html', image_url=IMAGE_PATH)
        except Exception as e:
            print(f"Lỗi khi tạo ảnh: {e}")
            return render_template('index.html', error_message=f"Đã xảy ra lỗi khi tạo ảnh: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Trong môi trường production, bạn nên dùng Gunicorn hoặc uWSGI thay vì debug=True
    app.run(host="0.0.0.0", port=port, debug=False)