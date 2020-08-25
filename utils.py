import os
import cv2
from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import json


def _load_model():
    """
    Hàm khởi tạo mô hình
    :return:
    """
    # Khởi tạo model
    model = ResNet50(weights='imagenet')
    print("Load model complete!")
    return model


# Resize ảnh
def _preprocess_image(img, shape):
    """
    Hàm tiền xử lý ảnh
    :param img:
    :param shape:
    :return:
    Ảnh với kích thước của tham số shape
    """
    img_rz = img.resize(shape)
    img_rz = img_to_array(img_rz)
    img_rz = np.expand_dims(img_rz, axis=0)
    return img_rz


# Encoding numpy to json
class NumpyEncoder(json.JSONEncoder):
    '''
    Encoding numpy into json
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def load_images_from_folder(folder):
    """
    Hàm đọc toàn bộ ảnh trong thư mục folder
    :param folder:
    :return:
    Danh sách ảnh dạng numpy có trong thư mục
    """
    # Khởi tạo biến chứa danh sách ảnh
    images = []
    for filename in os.listdir(folder):
        # Đọc từng ảnh
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # Thêm ảnh mới vào biến images
            images.append(img)
    return images