import cv2
from torchsummary import summary
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Network as Net

import matplotlib.pyplot as plt

minValue = 70
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
low_range = np.array([0, 70, 100])
upper_range = np.array([20, 200, 255])
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


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100, 400)
fontScale = 1
fontColor = (255, 125, 125)
lineType = 2

PATH = "trained_model"

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Sample2.mp4')

net = Net.Network()
net.load_state_dict(torch.load(PATH))
net.eval()


# def preprocess(img):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # cv2.imshow("HSV", hsv)
#     # cv2.waitKey(0)
#     # Apply skin color range
#     mask = cv2.inRange(hsv, low_range, upper_range)
#     mask = cv2.erode(mask, skinkernel, iterations=1)
#     mask = cv2.dilate(mask, skinkernel, iterations=1)
#     # blur
#     mask = cv2.GaussianBlur(mask, (15, 15), 1)
#     # bitwise and mask original frame
#     res = cv2.bitwise_and(img, img, mask=mask)
#     # color to grayscale
#     res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
#     return res


# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
large = np.ones((50, 50), np.uint8)
ret, frame = cap.read()
mapi = {0: 'STOP', 1: 'RIGHT', 2: 'LEFT', 3: 'Background'}
while(ret):
    # Capture frame-by-frame
    img = frame.copy()
    # img = fgbg.apply(frame)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, large)
    # img = cv2.GaussianBlur(img, (21, 21), 0)
    img = preprocess(img)
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(img, (50, 50))
    # gray = preprocess(gray)
    net_in = gray.flatten()
    net_in = net_in.reshape(1, 1, 50, 50)
    net_in = np.array(net_in)
    net_in = torch.FloatTensor(net_in)

    predictions = net(Variable(net_in))
    net_out = torch.max(predictions.data, 1)[1]
    confidence = torch.max(predictions.data, 1)[0]

    cv2.putText(frame, mapi[net_out.detach().numpy()[0]],
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.putText(frame, str(confidence.detach().numpy()[0]),
                (300, 400),
                font,
                fontScale,
                fontColor,
                lineType)

    # print(predictions.detach().numpy())
    print(str(net_out.detach().numpy()), str(confidence.detach().numpy()))
    cv2.waitKey(50)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
