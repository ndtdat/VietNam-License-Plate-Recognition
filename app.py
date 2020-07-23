import numpy as np
from flask import Flask, request
import json
import hyper as hp

import cv2
import io

import utils
from recognition import E2E

global model
model = None
# Khởi tạo flask app
app = Flask(__name__)

# Khai báo các route 1 cho API
@app.route("/")
# Khai báo hàm xử lý dữ liệu.
def _hello_world():
	return "Hello world"

# Khai báo các route 2 cho API
@app.route("/predict", methods=["POST"])
# Khai báo hàm xử lý dữ liệu.
def _predict():
	data = {"success": False}
	if request.files.get("image"):
		# Lấy file ảnh người dùng upload lên
		image = request.files["image"].read()
		# Convert sang dạng array image
		image_stream = io.BytesIO(image)
		file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
		img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
		# recognize license plate
		lpNumber = model.predict(img)
		data["license_plate"] = lpNumber
		data["success"] = True
	return json.dumps(data, ensure_ascii=False, cls=utils.NumpyEncoder)

if __name__ == "__main__":
	print("App run!")
	# Load model
	model = E2E()
	app.run(debug=False, host=hp.IP, threaded=False)