import cv2
from glob import glob
import os
import numpy as np
import pickle
from tqdm.auto import tqdm

train_path = '/Users/soyoung/ml_project/emotion_proj/data/train/'
test_path = '/Users/soyoung/ml_project/emotion_proj/data/test/'

class_list = os.listdir(train_path)

# Train 데이터 생성
train_dataset = []
noneset = []

for class_name in tqdm(class_list):
    train_files = glob(train_path+class_name+"/*.png")

    for train_file in train_files:
        img = cv2.imread(train_file)
        if img is None:
            noneset.append(train_file)
            continue
        train_data = {'image': img, 'label': class_name}
        train_dataset.append(train_data)

# Train 데이터 저장
with open('emo_train.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)

# Test 데이터 생성
test_dataset = []
noneset = []

for class_name in tqdm(class_list):
    test_files = glob(test_path+class_name+"/*.png")

    for test_file in test_files:
        img = cv2.imread(test_file)
        if img is None:
            noneset.append(test_file)
            continue
        test_data = {'image': img, 'label': class_name}
        test_dataset.append(test_data)

# Test 데이터 저장
with open('emo_test.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)
