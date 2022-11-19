# Program To Read video: https://github.com/shawnsky/extract-subtitles
# and Extract Frames
# Import required packages
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract\\tesseract.exe'
from imutils import contours

import imutils
# from imutils import contours
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import re
# from collections import Counter

def processMask(img,kernel):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sensitivity = 20
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])
    white = cv2.inRange(hsv,lower_white,upper_white)
    black = cv2.inRange(hsv,(0,0,0),(255,255,255))

    mask = white
    target = cv2.bitwise_and(img,img, mask=mask)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    if kernel == 'kernel':
        kernelSize = 3
        opIterations = 2
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
        target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    target = 255- target
    return target
def remove_noise(img,threshold):
    """
    remove salt-and-pepper noise in a binary image
    """
    filtered_img = np.zeros_like(img)
    labels,stats= cv2.connectedComponentsWithStats(img.astype(np.uint8),connectivity=8)[1:3]

    label_areas = stats[1:, cv2.CC_STAT_AREA]
    for i,label_area in enumerate(label_areas):
        if label_area > threshold:
            filtered_img[labels==i+1] = 1
    return filtered_img
def preprocess(img):
    """
    convert the grayscale captcha image to a clean binary image
    """
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (3,3), 0)

    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)[1]

    filtered_img = 255-remove_noise(thresh,5)*255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.erode(filtered_img,kernel,iterations = 1)
    return img

def size_threshold(img,minimum, maximum):
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    # print(stats[:, 4])
    for val in np.where((stats[:, 4] < minimum) + (stats[:, 4] > maximum))[0]:
      labels[labels==val] = 0
    return (labels > 0).astype(np.uint8) * 255

def extractTextAcc(path):
    img = cv2.imread(path)
    text = ''
    # x = 300
    # y = 640
    # w = 700
    # h = 50
    # img = img[y:y+h, x:x+w]

    # width = 1600
    # scaling = width/img.shape[1]
    # height = int(img.shape[0]*scaling)
    # resized_dimensions = (width, height)
    # img = cv2.resize(img, resized_dimensions,
    #                             interpolation=cv2.INTER_AREA)  # Shrinking images


    # img = processMask(img,'nokernel')
    # img = preprocess(img)
    # img = 255 - img
    # img = size_threshold(img, 20, 2000)  # after Mask+kernel
    # img = 255 - img
    # text=pytesseract.image_to_string(img, lang='chi_sim')

    hImg,wImg,_= img.shape
    # hImg,wImg,_ = img.shape
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ## c. Get contours and calc midpoint
    # cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # # sort the contours from left-to-right and initialize the
    # # 'pixels per metric' calibration variable
    # (cnts, _) = contours.sort_contours(cnts)
    # for i in range(0,len(cnts)):
    #     c = cnts[i]
    #     # if the contour is not sufficiently large, ignore it
    #     # if cv2.contourArea(c) < 100:
    #     #     continue
    #     # compute the center of the contour
    #     M = cv2.moments(c)
    #     cX = int(M["m10"] / M["m00"]) - 5 # adjust so it is not perfect in mid
    #     cY = int(M["m01"] / M["m00"])

    #     # draw the contour and center of the shape on the image
    #     cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    #     cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)

    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)

    boxes = pytesseract.image_to_boxes(img, lang='chi_sim',config='--psm 10')
    text = ""
    count = 0
    for b in boxes.splitlines():
        count+= 1
        if count == 1:
            b = b.split(' ')
            x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
            Y1 = hImg
            Y2 = 0
            # Y1 = hImg-y
            # Y2 = hImg-h
            # cv2.rectangle(img,(x,Y1),(w,Y2),(0,0,255),1)
            # print((x,Y1),(w,Y2))
            line_width = w-x
            height = Y1-Y2
            width = 75 #line_width/4
            num_chars = round(line_width/width)  # divide by char width
            for i in range(0,num_chars):
                x1 = x + (i*width)
                x2 = x1 + width
                y1 = Y1
                y2 = Y2
                # print((x1,y1),(x2,y2))
                cropped = img[y2:y1, x1:x2]
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
                char = pytesseract.image_to_string(cropped, lang='chi_sim',config='--psm 13')  # 8 and 13 are correct
                # cv2.imshow('crop',cropped)
                # cv2.waitKey(0)
                text += char

            # text += b[0]
    line = text.replace(" ", "")
    line = line.replace("\n", "")
    # temp = ''
    # for ch in line:
    #     if re.search(u'[\u4e00-\u9fff]', ch):
    #         temp += ch
    # line = temp
    print(line)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.imwrite(f'./output/{fname}_bb.jpg',img)
    return text

# get total number of frames
fname = 'fox_spirit_ep65_30'
path = f'./data/{fname}.jpg'
line = extractTextAcc(path)
# line = line.replace(" ", "")
# line = line.replace("\n", "")
# temp = ''
# for ch in line:
#     if re.search(u'[\u4e00-\u9fff]', ch):
#         temp += ch
# line = temp
# print(line)



