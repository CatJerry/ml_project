import cv2
import numpy as np
import os
import pickle

data = []
labels = []
train_path = '/Users/soyoung/ml_project/emotion_proj/data/train'
test_path = '/Users/soyoung/ml_project/emotion_proj/data/test'

for label in os.listdir(train_path):
    for img_file in os.listdir(os.path.join(train_path, label)):
        img_path = os.path.join(train_path, label, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 이미지를 1채널로 로드
        img = cv2.resize(img, (48, 48))  # 이미지를 48x48 크기로 조정
        data.append(img)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# 데이터를 pkl 파일로 저장
with open("train_data.pkl", "wb") as f:
    pickle.dump((data, labels), f)

data = []
labels = []
for label in os.listdir(test_path):
    for img_file in os.listdir(os.path.join(test_path, label)):
        img_path = os.path.join(test_path, label, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 이미지를 1채널로 로드
        img = cv2.resize(img, (48, 48))  # 이미지를 48x48 크기로 조정
        data.append(img)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# 데이터를 pkl 파일로 저장
with open("test_data.pkl", "wb") as f:
    pickle.dump((data, labels), f)
