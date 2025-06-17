
from flask import Flask, render_template, request
from create_logo import create_pipeline, text_2_img
import torch
import os # Đảm bảo đã import os
import time # Để thêm dấu thời gian vào tên file ảnh

# Khoi tao Flask app
app = Flask(__name__)

# Dinh nghia tham so
# IMAGE_DIR = "static" # Không cần thiết nếu bạn đã dùng IMAGE_PATH
# IMAGE_FILENAME = "output.jpg" # Sẽ tạo tên file động



pipeline = create_pipeline()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    else: # Xử lý yêu cầu POST
        user_input = request.form.get("prompt")

        if not user_input:
            # Xử lý trường hợp người dùng không nhập prompt
            return render_template('index.html', error_message="Vui lòng nhập mô tả logo để tạo ảnh.")

        # Thêm dấu thời gian vào tên file để tránh cache và ghi đè ảnh cũ
        timestamp = int(time.time())
        # Tạo đường dẫn đầy đủ cho file ảnh trong thư mục static
        unique_image_filename = f"output_{timestamp}.jpg"
        image_save_path = os.path.join("static", unique_image_filename)
        # Đường dẫn URL tương đối để hiển thị trên web
        image_url_for_web = f"/static/{unique_image_filename}"

        print(f"Start generating image for prompt: '{user_input}'.....")
        try:
            # Đảm bảo pipeline đã được khởi tạo
            if pipeline is None:
                return render_template('index.html', error_message="Lỗi: Pipeline AI chưa được khởi tạo. Vui lòng thử lại.")

            img = text_2_img(user_input, pipeline)
            print('Finish generating image...')
            img.save(image_save_path)
            print(f"Image saved to: {image_save_path}")

            # Trả về template với URL của ảnh đã tạo
            return render_template('index.html', image_url=image_url_for_web)

        except Exception as e:
            # Bắt và in lỗi nếu có vấn đề trong quá trình tạo hoặc lưu ảnh
            print(f"Error during image generation or saving: {e}")
            return render_template('index.html', error_message=f"Đã xảy ra lỗi khi tạo logo: {e}. Vui lòng thử lại với prompt khác.")
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Trong môi trường production, bạn nên dùng Gunicorn hoặc uWSGI thay vì debug=True
    app.run(host="0.0.0.0", port=port, debug=False)