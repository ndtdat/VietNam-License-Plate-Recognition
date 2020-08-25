import os
import pandas as pd
from recognition import E2E
import cv2 as cv2
import argparse
import utils



def get_arguments():
    """
    Hàm xử lý tham số truyền bằng command line
    :return:
    Đối tượng dạng danh sách {tham số: giá trị}
    """
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./images/1.jpg')

    return arg.parse_args()


# Lấy tham số truyền vào command line
args = get_arguments()

# Khởi động model
model = E2E()

# Đọc tất cả các ảnh test
all_imgs = utils.load_images_from_folder("D:/COMPUTER-VISION-PROJECT-WITH-CODE/Dataset/VNLP_test")

# # Thực nghiệm trên toàn bộ ảnh kiểm tra
# for i in range(len(all_imgs)):
#     # Dự đoán, trả về ảnh dự đoán và giá trị dự đoán
#     image, lpnumber = model.predict(all_imgs[i])
#
#     # In ra console để quan sát kết quả
#     print(lpnumber)
#
#     # Hiển thị ảnh dự đoán
#     cv2.imshow('Result', image)
#     cv2.waitKey(0)
folder = "D:/COMPUTER-VISION-PROJECT-WITH-CODE/Dataset/VNLP_test"
data = {
    'Filename': [],
    'Predict': []
}
for filename in os.listdir(folder):
    # Đọc từng ảnh
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
        # Dự đoán, trả về ảnh dự đoán và giá trị dự đoán
        image, lpNumber = model.predict(img)
        # In ra console để quan sát kết quả
        print(filename + ': ' + lpNumber)
        # Thêm ảnh mới vào biến images
        data['Filename'].append(filename)
        data['Predict'].append(lpNumber)

        image = None
        lpNumber = None

df = pd.DataFrame(data, columns=['Filename', 'Predict'])
df.to_csv('Predict.csv')
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)
cv2.destroyAllWindows()
