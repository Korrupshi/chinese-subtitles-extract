## This file is the full pipeline from video to subtitles: based on Current best Pipeline
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract\\tesseract.exe'
# import imutils
from imutils import contours
# from  PIL import  Image
import numpy as np
import re
# from collections import Counter

output = ''
def extractFrames(path):
    global output
    # global results
    cap = cv2.VideoCapture(path)
    count = 0
    frame = 0
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(1,totalFrames):
        count += 1
        success, img = cap.read()
        if count == totalFrames:
            break
        if count % 60 == 0:
            frame +=1
            if frame > 13:
                print(f'{frame}/{round(totalFrames/60)}')
                line =extractTextAcc(img,frame)
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                temp = ''
                for ch in line:
                    if re.search(u'[\u4e00-\u9fff]', ch):
                        temp += ch
                line = temp
                if line =="":
                    continue
                print(line)
                output += f'{line}\n'
                
                with open(f'./output/{fname}_subs.txt', 'w',encoding="utf-8") as f:
                    f.write(output)
        

def processMask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sensitivity = 20
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])
    white = cv2.inRange(hsv,lower_white,upper_white)
    black = cv2.inRange(hsv,(0,0,0),(255,255,255))

    mask = white
    target = cv2.bitwise_and(img,img, mask=mask)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
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
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    for val in np.where((stats[:, 4] < minimum) + (stats[:, 4] > maximum))[0]:
      labels[labels==val] = 0
    return (labels > 0).astype(np.uint8) * 255

def extractTextAcc(img,frame):
    global output
    text = ''
    height, width, channels = img.shape
    x = int(width*0.20)
    w = int(width*0.60)
    y = int(height*0.87)
    h = 70
    img = img[y:y+h, x:x+w]

    width = 1600
    scaling = width/img.shape[1]
    height = int(img.shape[0]*scaling)
    resized_dimensions = (width, height)
    img = cv2.resize(img, resized_dimensions,
                                interpolation=cv2.INTER_AREA)
    img = processMask(img)
    img = preprocess(img)
    img = 255 - img
    img = size_threshold(img, 20, 2000)  # after Mask+kernel
    img = 255 - img

    ## > Checkpoint 1: Test if horizontal line has pixels, to save time
    thickness = 2
    xline = img[round(height/2)-thickness:round(height/2)+thickness, 0:width]  # hLine in mid
    testImg = cv2.bitwise_not(xline)  # convert to single channel
    if cv2.countNonZero(testImg) == 0 : # test if box contains character
        return ''

    hImg,wImg= img.shape
    img2 = img.copy()
    boxes = pytesseract.image_to_boxes(img, lang='chi_sim',config='--psm 7')  # 7 or 13
    ## > Checkpoint 2: Is there a box? Else return
    try:
        b = boxes.splitlines()[0]
        b = b.split(' ')
    except:
       return ''
    x1 = int(b[1])-5
    text = ""
    width = 75
    length = round(wImg/width)
    for i in range(0,length):
        y1 = hImg
        y2 = 0
        if i != 0:
            x1 += width
        x2 = x1 + width
        cropped = img[y2:y1, x1:x2]

        # > Test if block contains character
        # compute the center of the contour
        M = cv2.moments(cropped)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            continue
        
        thickness = 2
        xChar = cropped[cY-thickness:cY+thickness, 0:width]  # hLine in mid
        yChar = cropped[y2:y1, cX-thickness:cX+thickness]  # vline in mid
        testX = cv2.bitwise_not(xChar)  # convert to single channel
        testY = cv2.bitwise_not(yChar)  # convert to single channel
        if cv2.countNonZero(testX) != 0 and cv2.countNonZero(testY) != 0: # test if box contains character
                cv2.rectangle(img2,(x1,y1),(x2,y2),(0,0,255),1)
                char = pytesseract.image_to_string(cropped, lang='chi_sim',config='--psm 8')  # 8 and 13 are correct
                text += char
    return text

fname = 'fox_spirit_ep65'
path = f'./data/{fname}.flv'

extractFrames(path)


