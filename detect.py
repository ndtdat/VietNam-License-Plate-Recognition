import data_utils as utils
import cv2
import numpy as np

# Class phát hiện vùng chứa biển số sử dụng YOLOv3 Tiny
class detectNumberPlate(object):
    def __init__(self, threshold=0.5):
        args = utils.get_arguments()
        self.weight_path = args.weight_path
        self.cfg_path = args.config_path
        self.labels = utils.get_labels(args.classes_path)
        self.threshold = threshold

        # Tải mô hình YOLOv3 Tiny cho phát hiện vùng biển số
        self.model = cv2.dnn.readNet(model=self.weight_path, config=self.cfg_path)

    def detect(self, image):
        boxes = []
        classes_id = []
        confidences = []
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scalefactor=scale, size=(416, 416), mean=(0, 0), swapRB=True, crop=False)
        height, width = image.shape[:2]

        # Lấy ảnh đầu vào cho model
        self.model.setInput(blob)

        # Thực hiện tính toán forward
        outputs = self.model.forward(utils.get_output_layers(self.model))

        # Xử lý các vùng có khả năng là biển số xe
        for output in outputs:
            for i in range(len(output)):
                scores = output[i][5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])

                # Chọn vùng có độ tin cậy rằng vùng đó khả năng lớn là biển số
                if confidence > self.threshold:
                    # coordinate of bounding boxes
                    center_x = int(output[i][0] * width)
                    center_y = int(output[i][1] * height)

                    detected_width = int(output[i][2] * width)
                    detected_height = int(output[i][3] * height)

                    x_min = center_x - detected_width / 2
                    y_min = center_y - detected_height / 2

                    boxes.append([x_min, y_min, detected_width, detected_height])
                    classes_id.append(class_id)
                    confidences.append(confidence)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.threshold, nms_threshold=0.4)

        coordinates = []
        # Lưu giá trị của các vùng chắc chắn là biển số
        for i in indices:
            index = i[0]
            x_min, y_min, width, height = boxes[index]
            x_min = round(x_min)
            y_min = round(y_min)

            coordinates.append((x_min, y_min, width, height))

        return coordinates
