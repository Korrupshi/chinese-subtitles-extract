## This file is the full pipeline from video to subtitles: based on Current best Pipeline
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract\\tesseract.exe'
# import imutils
from imutils import contours
# from  PIL import  Image
import numpy as np
import re
import time
# from collections import Counter
from PIL import Image, ImageEnhance, ImageFilter
output = ''
def extractFrames(path):
    start = time.time()

    global output
    # global results
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 24 fps
    # res = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # resolution
    # print(f'{res}p, {fps}FPS')
    # return
    fps = 30
    count = 0
    frame = 0
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(1,totalFrames):
        count += 1
        success, img = cap.read()
        if count == totalFrames:
            break
        if count % fps*2 == 0:
            frame +=1
            if frame >30:
                print(f'{frame}/{round(totalFrames/(fps*2))}')
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
                
                # with open(f'./output/{fname}_subs.txt', 'w',encoding="utf-8") as f:
                #     f.write(output)
    
    end = time.time()
    # get the execution time
    elapsed_time = end - start
    print(f'> {len(output.splitlines())} lines in {round(elapsed_time/60)}min')

        

def processMask(img):

    # > Increase contrast
    img = Image.fromarray(img)
    img = ImageEnhance.Contrast(img)
    img = img.enhance(1.20)
    img = np.array(img, dtype= "uint8")

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sensitivity = 0  # or 12  | 20
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([0,sensitivity,255])
    # lower_black = np.array([0, 0, 0])
    # upper_black = np.array([350,55,100])
    white = cv2.inRange(hsv,lower_white,upper_white)
    # black = cv2.inRange(hsv,lower_black,upper_black)
    mask = white
    target = cv2.bitwise_and(img,img, mask=mask)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # target = 255- target

    # > convert to LAB 1
    # I_LAB = cv2.cvtColor(np.array(img,dtype="uint8"), cv2.COLOR_RGB2LAB)
    # L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
    # intensity = 0.05
    # dark = cv2.inRange(L,0,0+intensity)
    # light = cv2.inRange(L,1-intensity,1)
    # mask = dark | light
    # target = cv2.bitwise_and(img,img, mask=mask)
    # target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # temp = 255 - target
    # cv2.imshow('lab',temp)
    # cv2.waitKey(0)
    # new_image = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    # target.show()
    return target

def Lab_Segmentation(image):
    lowerRange= np.array([0, 135, 135] , dtype="uint8")
    upperRange= np.array([255, 160, 195], dtype="uint8")
    mask = image[:].copy()

    imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    imageRange = cv2.inRange(imageLab,lowerRange, upperRange)
    
    mask[:,:,0] = imageRange
    mask[:,:,1] = imageRange
    mask[:,:,2] = imageRange
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    faceLab = cv2.bitwise_and(image,mask)

    cv2.imshow('lab',faceLab)
    cv2.waitKey(0)

    return faceLab

def loopGrey(img):
    # > Increase contrast
    img = Image.fromarray(img)
    img = ImageEnhance.Contrast(img)
    img = img.enhance(1.20)
    img = np.array(img, dtype= "uint8")
    masks = []
    shades = 80
    for i in range(0,shades):
        a = 255 - i
        min_grey = np.array([a,a,a])
        mask = cv2.inRange(img,min_grey,min_grey)
        masks.append(mask)
    
    count = 0
    for mask in masks:
        count += 1
        target = cv2.bitwise_and(img,img, mask=mask)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        if count == 1:
            result = target
        result = cv2.add(result, target)
    return result        

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
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (3,3), 0)

    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)[1]

    filtered_img = 255-remove_noise(thresh,5)*255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.erode(filtered_img,kernel,iterations = 1)
    return img


def size_threshold(img,minimum, maximum):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    # print(stats[:, 4])
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

    # Increase contrast
    # # > Increase contrast
    # img = Image.fromarray(img)
    # img = ImageEnhance.Contrast(img)
    # img = img.enhance(1.2)
    # img = np.array(img, dtype= "uint8")
    # lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # l_channel, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(8,8))
    # cl = clahe.apply(l_channel)
    # limg = cv2.merge((cl,a,b))
    # img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # img = np.hstack((img, img))

    # img3 = Lab_Segmentation(img)
    # img = processMask(img)
    img = loopGrey(img)
    
    img = 255 - img
    img = preprocess(img)
    img = 255 - img
    img = size_threshold(img, 20, 2500)  # after Mask+kernel 20-2000
    img = 255 - img


    ## > Checkpoint 1: Test if horizontal line has pixels, to save time
    thickness = 4
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
        # else:
        #     cv2.imshow('img',cropped)
        #     cv2.waitKey(0) 

    cv2.imshow('img',img2)
    cv2.waitKey(0)          
    return text

fname = 'fox_spirit_ep65'
path = f'./data/{fname}.flv'

# get the start time
extractFrames(path)
