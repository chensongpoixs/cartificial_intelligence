


import os;
import cv2;
from PIL import Image;
import numpy as np;


def getImageAndLables(path):
    #存储人脸数据
    facessSamples = [];
    #存储姓名数据
    ids = [];
    # 存储图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)];
    # 加载分类器
    face_detector = cv2.CascadeClassifier("opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
    #遍历列表中的图片
    for imagePath in imagePaths:
        # 打开图片， 灰度化 PIL有九中不同格式: 1， L， P, RGB, RGBA, CMYK , YCbCr, I, F
        PIL_img = Image.open(imagePath).convert('L');
        # 将图像转换为数组， 以黑白深浅
        img_numpy = np.array(PIL_img, 'uint8');
        # 获取图片人脸体征
        faces = face_detector.detectMultiScale(img_numpy);
        # 获取每张图片的id和姓名
        id = int(os.path.split(imagePath)[1].split('.')[0]);
        #预防五面容照片
        for x, y, w, h in faces:
            ids.append(id);
            facessSamples.append(img_numpy[y:y+h, x:x+w]);


    print("id:", id);
    print("fs:", facessSamples);
    return facessSamples, ids;


if __name__ == '__main__':
    #图片路径
    path = "./data/";
    #获取图像数组和id标签数组和姓名
    facess, ids = getImageAndLables(path);
    # 加载识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create();

    #
    recognizer.train(faces, np.array(ids));

    # 保存文件
    recognizer.write("trainer/trainer.yml");

