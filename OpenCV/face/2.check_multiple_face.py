# auther: chensong
# date: 2023-05-30

import  cv2 as cv;


FACE_TRAN_DATA_PATH = "D:/Work/cartificial_intelligence/OpenCV/opencv/data/haarcascades/haarcascade_frontalface_default.xml";

# 检查函数
def face_detect_demo():

    gary = cv.cvtColor(img, cv.COLOR_BGR2GRAY);

    face_detect = cv.CascadeClassifier(FACE_TRAN_DATA_PATH);

    # TODO@chensong 2023-05-30  这个里面参数还是有点意思啦
    face = face_detect.detectMultiScale(gary, 1.01, 5, 0, (5, 5) , (300, 300));

    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color = (0, 0, 255), thickness = 2);
    cv.imshow("result", img);


img = cv.imread("face.jpg");

face_detect_demo();

while True:
    if ord('q') == cv.waitKey(0):
        break;

cv.destroyAllWindows();