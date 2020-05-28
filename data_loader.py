import cv2
import numpy as np
import os
import csv
import sys
from PIL import Image
import random

minValue = 70
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
low_range = np.array([0, 70, 100])
upper_range = np.array([20, 200, 255])


# cv2.imshow("final",img)
# cv2.waitKey(0)
kernel = np.ones((5, 5), np.uint8)


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # edged = cv2.Canny(gray, 30, 90)
    # th3 = cv2.adaptiveThreshold(
    #     blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, thresh1 = cv2.threshold(
        blur, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh1


# th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
# ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# cv2.imshow("final",blur)
# cv2.waitKey(0)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# #Apply skin color range
# mask = cv2.inRange(hsv, low_range, upper_range)
# mask = cv2.erode(mask, skinkernel, iterations = 1)
# mask = cv2.dilate(mask, skinkernel, iterations = 1)
# #blur
# mask = cv2.GaussianBlur(mask, (15,15), 1)
# #cv2.imshow("Blur", mask)
# #bitwise and mask original frame
# res = cv2.bitwise_and(img, img, mask = mask)
# # color to grayscale
# res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
# cv2.imshow("final",res)
# cv2.waitKey(0)

# Useful function
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


# load the original image
myFileList = createFileList('Dataset/ARROW_LEFT')

val = []
val.append("label")
for i in range(1, 2501):
    val.append("pixel"+str(i))

with open("train.csv", 'a') as f:
    writer = csv.writer(f)
    writer.writerow(val)

with open("validation.csv", 'a') as f:
    writer = csv.writer(f)
    writer.writerow(val)

with open("test.csv", 'a') as f:
    writer = csv.writer(f)
    writer.writerow(val)


train = []
test = []
validation = []

myFileList = createFileList('Dataset/ARROW_LEFT')
ctr = 0
for file in myFileList:
    print(file)
    img_file = cv2.imread(file)
    width, height = 50, 50
    img_grey = preprocess(img_file)
    value = img_grey.flatten()
    ctr = ctr + 1
    value = np.insert(value, 0, 1)

    if(ctr <= 6):
        print("Train")
        train.append(value)
    elif(ctr <= 8):
        print('Validation')
        validation.append(value)
    else:
        print("Test")
        test.append(value)
        if(ctr == 10):
            ctr = 0

myFileList = createFileList('Dataset/ARROW_RIGHT')
ctr = 0
for file in myFileList:
    print(file)
    img_file = cv2.imread(file)
    width, height = 50, 50
    img_grey = preprocess(img_file)
    value = img_grey.flatten()
    ctr = ctr + 1

    value = np.insert(value, 0, 2)

    if(ctr <= 6):
        print("Train")
        train.append(value)
    elif(ctr <= 8):
        print("Validation")
        validation.append(value)
    else:
        print("Test")
        test.append(value)
        if(ctr == 10):
            ctr = 0


myFileList = createFileList('Dataset/STOP')
ctr = 0
for file in myFileList:
    print(file)
    img_file = cv2.imread(file)
    width, height = 50, 50
    img_grey = preprocess(img_file)
    value = img_grey.flatten()
    ctr = ctr + 1

    value = np.insert(value, 0, 0)

    if(ctr <= 6):
        print("Train")
        train.append(value)
    elif(ctr <= 8):
        print("Validation")
        validation.append(value)
    else:
        print("Test")
        test.append(value)
        if(ctr == 10):
            ctr = 0

myFileList = createFileList('Dataset/PLAIN')
ctr = 0
for file in myFileList:
    print(file)
    img_file = cv2.imread(file)
    width, height = 50, 50
    img_grey = preprocess(img_file)
    value = img_grey.flatten()
    ctr = ctr + 1

    value = np.insert(value, 0, 3)

    if(ctr <= 6):
        print("Train")
        train.append(value)
    elif(ctr <= 8):
        print("Validation")
        validation.append(value)
    else:
        print("Test")
        test.append(value)
        if(ctr == 10):
            ctr = 0

random.shuffle(train)
random.shuffle(validation)
random.shuffle(test)

print('Train')
for i in range(0, len(train)):
    with open("train.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(train[i])

print('VALID')
for i in range(0, len(validation)):
    with open("validation.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(validation[i])

print('TEST')
for i in range(0, len(test)):
    with open("test.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(test[i])
