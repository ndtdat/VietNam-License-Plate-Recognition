import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils
from data_utils import order_points, convert2Square, draw_labels_and_boxes
from detect import detectNumberPlate
from model import CNN_Model
from skimage.filters import threshold_local

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


class E2E(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        self.detectLP = detectNumberPlate()
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights('./weights/weight.h5')
        self.candidates = []
        self.lpNumber = None

    def extractLP(self):
        """
        Hàm trích xuất vùng chứa biển số xe
        :return:
        """
        coordinates = self.detectLP.detect(self.image)
        if len(coordinates) == 0:
            ValueError('No images detected')

        for coordinate in coordinates:
            yield coordinate

    def predict(self, image):
        """
        Hàm dự đoán giá trị của biển số xe
        :param image:
        :return:
        Ảnh được chú thích vùng biển số và giá trị
        Chuỗi ký tự biển số xe
        """
        # Ảnh đầu vào
        self.image = image

        # Xét các vùng biển số detect được bằng YOLOv3 Tiny
        for coordinate in self.extractLP():
            # Khởi tạo candidates để lưu giá trị biển số và tọa độ cần chú thích trong ảnh
            self.candidates = []

            # Chuyển đổi (x_min, y_min, width, height) thành dạng (top left, top right, bottom left, bottom right)
            pts = order_points(coordinate)

            # Cắt ảnh biển số xe dùng bird's eyes view transformation
            LpRegion = perspective.four_point_transform(self.image, pts)

            # Xử lý trường hợp biển số 1 dòng và 2 dòng
            # Chọn ngưỡng tỷ lệ chiều ngang / chiều dọc là 1.5
            # Nếu tỷ lệ này > 1.5 => Biển số 1 dòng
            # Ngược lại => Biển số 2 dòng
            if (LpRegion.shape[1]/LpRegion.shape[0] > 1.5):
                # Tỷ lệ scale
                scale_ratio = 40/LpRegion.shape[0]
                (w, h) = (int(LpRegion.shape[1]*scale_ratio), int(LpRegion.shape[0]*scale_ratio))
            else:
                # Tỷ lệ scale
                scale_ratio = 100 / LpRegion.shape[0]
                (w, h) = (int(LpRegion.shape[1] * scale_ratio), int(LpRegion.shape[0] * scale_ratio))

            # Resize ảnh vùng biển số về kích thước chuẩn
            # Đối với biển số 2 dòng: chiều cao = 40px
            # Đối với biển số 1 dòng: chiều cao = 100px
            LpRegion = cv2.resize(LpRegion, (w, h))

            # Phân đoạn từng ký tự
            self.segmentation(LpRegion)

            # Nhận diện các ký tự
            self.recognizeChar()

            # Định dạng các ký tự biển số
            self.lpNumber = self.format()

            # Vẽ bounding box và giá trị biển số vào ảnh
            self.image = draw_labels_and_boxes(self.image, self.lpNumber, coordinate)

        # Trả về ảnh dự đoán và giá trị biển số xe
        return self.image, self.lpNumber


    def segmentation(self, LpRegion):
        """
        Hàm phân đoạn ảnh
        :param LpRegion:
        :return:
        """
        # Áp dụng thresh để trích xuất vùng biển số
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]

        # Phân ngưỡng bằng adaptive threshold
        retval, threshold = cv2.threshold(V, 128, 255, cv2.THRESH_BINARY)
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255

        # Chuyển đổi pixel đen của chữ số thành pixel trắng
        thresh = cv2.bitwise_not(thresh)

        # Resize ảnh thresh với chiều rộng = 400px
        thresh = imutils.resize(thresh, width=400)

        # Xóa nhiễu bằng thuật toán opening (erode => dilate)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.cv2.erode(thresh, kernel)
        thresh = cv2.cv2.dilate(thresh, kernel)

        # Thực hiện thuật toán connected components analysis
        labels = measure.label(thresh, connectivity=2, background=0)

        # Lặp qua các thành phần duy nhất
        for label in np.unique(labels):
            # if this is background label, ignore it
            if label == 0:
                continue

            # Khởi tạo mặt nạ chứa vị trí của các ký tự ứng viên
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            # Tìm contours từ mặt nạ
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                # Xác định ký tự
                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                    # Trích xuất các ký tự
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    self.candidates.append((square_candidate, (y, x)))

    def recognizeChar(self):
        """
        Hàm nhận diện ký tự biển số xe
        :return:
        """
        # Khởi tạo danh sách ký tự và tọa độ của chúng
        characters = []
        coordinates = []

        # Gán giá trị cho characters và coordinates từ biến candidates
        for char, coordinate in self.candidates:
            characters.append(char)
            coordinates.append(coordinate)

        characters = np.array(characters)

        # Gán candidates là mảng empty
        self.candidates = []

        # Duyệt các ký tự ứng viên của vùng ảnh biển số đang xét
        if len(characters):
            result = self.recogChar.predict_on_batch(characters)
            result_idx = np.argmax(result, axis=1)

            for i in range(len(result_idx)):
                if result_idx[i] == 31:    # Bỏ qua trường hợp background
                    continue

                # Gán giá trị ký tự đã nhận diện được vào biến candidates
                self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    def format(self):
        """
        Hàm định dạng lại chuỗi ký tự biển số xe
        :return:
        """
        # Khởi tạo biến chứa các ký tự ở dòng 1 và dòng 2
        first_line = []
        second_line = []

        # Xác định ký tự trên từng dòng
        for candidate, coordinate in self.candidates:
            # Trường hợp biển số 1 dòng
            if self.candidates[0][1][0] + 40 > coordinate[0]:
                first_line.append((candidate, coordinate[1]))
            # Trường hợp biển số 2 dòng
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        # Gán giá trị cho chuỗi kết quả cuối cùng
        if len(second_line) == 0:  # if license plate has 1 line
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:   # if license plate has 2 lines
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "".join([str(ele[0]) for ele in second_line])

        return license_plate
