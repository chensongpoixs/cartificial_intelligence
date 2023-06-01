



import cv2
import numpy as np
import os
# coding=utf-8
import urllib
import urllib.request
import hashlib




FACE_TRAN_DATA_PATH = "D:/Work/cartificial_intelligence/OpenCV/opencv/data/haarcascades/haarcascade_frontalface_default.xml";

VIDEO_PATH = "C:/video/QingdaoUniversity.mp4";

#加载训练数据集文件
# 当然在运行之前有进行过pip install opencv-contrib-python，但是！一运行程序，会产生这样的报错recognizer=cv2.face.LBPHFaceRecognizer_create()
# AttributeError: module 'cv2' has no attribute 'face' 常见的办法就是 卸载、

recogizer=cv2.face.LBPHFaceRecognizer_create();
recogizer.read('trainer/trainer.yml')
names=[]
warningtime = 0

def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

statusStr = {
    '0': '短信发送成功',
    '-1': '参数不全',
    '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
    '30': '密码错误',
    '40': '账号不存在',
    '41': '余额不足',
    '42': '账户已过期',
    '43': 'IP地址限制',
    '50': '内容含有敏感词'
}


def warning():
    smsapi = "http://api.smsbao.com/"
    # 短信平台账号
    user = '13******10'
    # 短信平台密码
    password = md5('*******')
    # 要发送的短信内容
    content = '【报警】\n原因：检测到未知人员\n地点：xxx'
    # 要发送短信的手机号码
    phone = '*******'

    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print(statusStr[the_page])

#准备识别的图片
def face_detect_demo(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度
    face_detector=cv2.CascadeClassifier(FACE_TRAN_DATA_PATH)
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(2,2),(300,300))
    #face=face_detector.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
        # 人脸识别
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        print('标签id:',ids,'置信评分：', confidence)
        if confidence > 80:
            global warningtime
            warningtime += 1
            if warningtime > 100:
               warning()
               warningtime = 0
            cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.putText(img,str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result',img)
    #print('bug:',ids)

def name():
    path = './img'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name)


cap=cv2.VideoCapture(VIDEO_PATH)
name()
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):
        break
cv2.destroyAllWindows()
cap.release()
#print(names)
