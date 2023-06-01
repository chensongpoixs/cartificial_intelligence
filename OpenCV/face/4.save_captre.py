
# auther: chensong
# date: 2023-05-30

#

import cv2;

cap = cv2.VideoCapture(0);


falg = 1;
num = 1;

while (cap.isOpened()):
    ret_flag, Vshow = cap.read();
    cv2.imshow("Capture_test", Vshow);
    k = cv2.waitKey(1) & 0XFF;

    if  k == ord('s'):
        cv2.imwrite("./img/" + str(num)+"_name.jpg", Vshow);
        print('success to save ' + str(num) + '_name.jpg');
        print("----------------");
        num += 1;
    elif k == ord(' '):
        break;

cap.release();

cv2.destroyAllWindows();
