import cv2
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 
from torchvision import models


device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

#Hyper parameter 설정 
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

def model_load():
    # 모델 불러오기
    model = models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=7)
    model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # pre trained model
    model.load_state_dict(torch.load("/Users/soyoung/ml_project/emotion_proj/4/best_model.pt"))
    model.to(device)
    return model
pre_model = model_load()

def model_pred(model,img):

    pred = torch.stack([torch.tensor(im, dtype=torch.float32) for im in img])  # (N, 48, 48)
    pred = pred.permute(0, 3, 1, 2)
    pred_loader = DataLoader(pred, batch_size = BATCH_SIZE, shuffle=True) #drop_last = True

    for epoch in range(EPOCHS):
        with torch.no_grad():
            for img in pred_loader:
                pred = model(img.to(device))

                _, predicted = torch.max(pred.data, 1)  # 예측된 라벨
             # clear_output()
            if predicted==3:
                print("happy")
            elif predicted==4:
                print("neutral")


#웹캠에서 영상을 읽어온다
cap = cv2.VideoCapture(0)
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

#얼굴 인식 캐스케이드 파일 읽는다
face_cascade = cv2.CascadeClassifier('/Users/soyoung/ml_project/emotion_proj/3/haarcascade_frontalface_alt.xml')
count = 0

def webcam(cap, face_cascade, count):
    model_input_size = (48, 48) # 모델에 입력되는 이미지 크기
    while(True):
    # frame 별로 capture 한다
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #인식된 얼굴 갯수를 출력
        print(len(faces))

    # 인식된 얼굴에 사각형을 출력한다
        for (x,y,w,h) in faces:
             cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

          # 추출된 얼굴 영역 이미지를 모델에 입력할 준비
             face_img = gray[y:y+h, x:x+w]
             face_img = cv2.resize(face_img, model_input_size)
             face_img = np.expand_dims(face_img, axis=0)
         # 모델에 입력되는 이미지는 일반적으로 채널 수가 3이기 때문에 채널 수를 1로 바꿔줌
             face_img = np.expand_dims(face_img, axis=-1) 

         # 모델에 입력하여 결과 출력
             model_pred(pre_model,face_img)
         

    #화면에 출력한다
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


webcam(cap, face_cascade, count)

  

