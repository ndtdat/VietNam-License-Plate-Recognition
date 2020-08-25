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

# Thực nghiệm trên toàn bộ ảnh kiểm tra
for i in range(len(all_imgs)):
    # Dự đoán, trả về ảnh dự đoán và giá trị dự đoán
    image, lpnumber = model.predict(all_imgs[i])

    # In ra console để quan sát kết quả
    print(lpnumber)

    # Hiển thị ảnh dự đoán
    cv2.imshow('Result', image)
    cv2.waitKey(0)

if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)
cv2.destroyAllWindows()
