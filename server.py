import numpy as np
from flask import Flask, request
import json
from flask_cors import CORS
import hyper as hp
import cv2
import io
import base64
import utils
from recognition import E2E

global model
model = None
# Khởi tạo flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Khai báo các route 1 cho API
@app.route("/")
# Khai báo hàm xử lý dữ liệu.
def _hello_world():
    return "Hello world"


# Khai báo các route 2 cho API
@app.route("/predict", methods=["POST"])
# Khai báo hàm xử lý dữ liệu.
def predict():
    data = {"success": False}
    file_name = request.form.get("filename")
    print(file_name)
    if request.files.get("image"):
        # Lấy file ảnh người dùng upload lên
        image = request.files["image"].read()
        # Convert sang dạng array image
        image_stream = io.BytesIO(image)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # recognize license plate

        img, lpNumber = model.predict(img)
        file_name = "image_predict/pred_" + file_name
        cv2.imwrite(file_name, img)

        # convert image to byte type
        with open(file_name, "rb") as image_file:
            encoded_string1 = base64.b64encode(image_file.read())

        # convert byte to string
        encoded_string = encoded_string1.decode("utf-8")

        data["image"] = encoded_string
        data["license_plate"] = lpNumber
        data["success"] = True
    return json.dumps(data, ensure_ascii=False, cls=utils.NumpyEncoder)


if __name__ == "__main__":
    print("App run!")
    # Load model
    model = E2E()
    app.run(debug=False, host=hp.IP, threaded=False)
