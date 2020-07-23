# VietNam License Plate Recognition
Phương pháp đề xuất bao gồm 4 bước: <br>

* Xác định vùng chứa biển số xe sử dụng **Yolo Tiny v3** 
* Sử dụng thuật toán segment để tách từng kí tự trên biển số xe
* Xây dựng một model CNN để phân loại các kí tự(characters classification)
* Định dạng lại biển số xe xác định biển số xe gồm một hay hai dòng.

## Thử nghiệm
* Thực thi bằng câu lệnh dưới đây, lưu ý thay thế link_to_image bằng đường dẫn tới ảnh muốn đọc. <br>
* Hiện tại, source code chỉ thực thi trên ảnh tĩnh.
```
python test.py --image_path=link_to_image 
```

## Dependencies
* python==3.6
* tensorflow-cpu==2.2.0
* keras==2.3.1
* numpy==1.18.5
* opencv==4.3.0.36
* scikit-image==0.16.2
* imutils==0.5.3

## Tham khảo
https://viblo.asia/p/nhan-dien-bien-so-xe-viet-nam-Do754P9L5M6