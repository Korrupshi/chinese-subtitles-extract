# Program To Read video: https://github.com/shawnsky/extract-subtitles
# and Extract Frames
# Import required packages
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract\\tesseract.exe'
import imutils
from imutils import contours
from  PIL import  Image
import numpy as np
import re
from collections import Counter

results = []
output = f'size\tprocessing\taccuracy\n'
def extractFrames(path):
    global output
    global results
    cap = cv2.VideoCapture(path)
    count = 0
    # success = 1
    frame = 0
    # output = ''
    text = ''
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(size,m)
    # while True:
    for i in range(1,totalFrames):
        # print(i)
        count += 1
        success, img = cap.read()
        if count == totalFrames:
            break
        if count % 60 == 0:
            frame +=1
            # print(f'{frame}/{round(totalFrames/60)}')
            
            if frame > 10:
                line =extractTextAcc(img,frame,size,m)
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                temp = ''
                for ch in line:
                    if re.search(u'[\u4e00-\u9fff]', ch):
                        temp += ch
                line = temp
                if line =="":
                    continue
                results.append(line)
                if len(results) == 13:
                    row = f'{size}\t{m}\t'
                    corr = 0
                    total = 0
                    for l in range(0,len(results)):
                        # for ch in results[l]:
                        ans = data[l]
                        test = results[l]
                        # print(ans,test)
                        correct = Counter(test) & Counter(ans)  # => {'q': 2, 'r': 1}
                        corr += sum(correct.values())
                        total += len(ans)
                    acc = round(corr/total,2)
                    row+= f'{acc}\n'
                    output += row
                            
                    print(output)
                    with open('./output/accuracy2.txt', 'w') as f:
                        f.write(output)
                    results = []
                    break
            
            if frame > 40:
                row = f'{size}\t{m}\t'
                corr = 0
                total = 0
                for l in range(0,len(results)):
                    # for ch in results[l]:
                    ans = data[l]
                    test = results[l]
                    # print(ans,test)
                    correct = Counter(test) & Counter(ans)  # => {'q': 2, 'r': 1}
                    corr += sum(correct.values())
                    total += len(ans)
                acc = round(corr/total,2)
                row+= f'{acc}\n'
                output += row
                        
                print(output)
                with open('./output/accuracy2.txt', 'w') as f:
                    f.write(output)
                results = []
                break

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
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (3,3), 0)

    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)[1]

    filtered_img = 255-remove_noise(thresh,20)*255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.erode(filtered_img,kernel,iterations = 1)
    # img = 255 -img
    # width = 700
    # scaling = width/img.shape[1]
    # height = int(img.shape[0]*scaling)
    # resized_dimensions = (width, height)
    # img = cv2.resize(img, resized_dimensions,
    #                             interpolation=cv2.INTER_AREA)  # Shrinking images
    return img

def extract_letters(img):
    text = pytesseract.image_to_string(img,lang='chi_sim')#,config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    return text

def processMask(img):
    # img = 255-img
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sensitivity = 20
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])
    white = cv2.inRange(hsv,lower_white,upper_white)
    black = cv2.inRange(hsv,(0,0,0),(255,255,255))

    mask = white
    target = cv2.bitwise_and(img,img, mask=mask)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('t',target)
    # cv2.waitKey(0)

    return target

## Process method 2: Binary
def processBi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw

def areaFilter(minArea, inputImage):
    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(inputImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage

def processBlack(img):
    # https://stackoverflow.com/questions/70300189/how-to-keep-only-black-color-text-in-the-image-using-opencv-python
    # Conversion to CMYK (just the K channel black):
    # Convert to float and divide by 255:
    imgFloat = img.astype(float) / 255

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)
    # Threshold image:
    binaryThresh = 190
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)

    # cv2.imshow("binaryImage", binaryImage)
    # cv2.waitKey(0)

    # Filter small blobs:
    # minArea = 100
    # binaryImage = areaFilter(minArea, binaryImage)
    # cv2.imshow("binaryImage", binaryImage)
    # cv2.waitKey(0)
    # Use a little bit of morphology to clean the mask:
    kernelSize = 3
    opIterations = 2
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    return binaryImage

def extractTextAcc(img,frame,size,method):
    text = ''
    x = 300
    y = 630
    w = 700
    h = 50
    img = img[y:y+h, x:x+w]

    if size != 'none':
    # resize
        width = 1600
        scaling = width/img.shape[1]
        height = int(img.shape[0]*scaling)
        resized_dimensions = (width, height)
        img = cv2.resize(img, resized_dimensions,
                                    interpolation=cv2.INTER_AREA)
    if method == "none":
        text=pytesseract.image_to_string(img, lang='chi_sim')
    if method == 'whiteMask':
        img = processMask(img)
        # width = 700
        # scaling = width/img.shape[1]
        # height = int(img.shape[0]*scaling)
        # resized_dimensions = (width, height)
        # img = cv2.resize(img, resized_dimensions,
        #                             interpolation=cv2.INTER_AREA)
        text=pytesseract.image_to_string(img, lang='chi_sim')
    if method == 'biThresh':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        width = 700
        scaling = width/img.shape[1]
        height = int(img.shape[0]*scaling)
        resized_dimensions = (width, height)
        img = cv2.resize(img, resized_dimensions,
                                    interpolation=cv2.INTER_AREA)
        text=pytesseract.image_to_string(img, lang='chi_sim')
    if method == 'kernel':
        img = processBlack(img)
        width = 700
        scaling = width/img.shape[1]
        height = int(img.shape[0]*scaling)
        resized_dimensions = (width, height)
        img = cv2.resize(img, resized_dimensions,
                                    interpolation=cv2.INTER_AREA)
        text=pytesseract.image_to_string(img, lang='chi_sim')
    if method == 'whiteMask+kernel':
        img = processMask(img)
        kernelSize = 3
        opIterations = 2
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
        img = 255- img
        # width = 700
        # scaling = width/img.shape[1]
        # height = int(img.shape[0]*scaling)
        # resized_dimensions = (width, height)
        # img = cv2.resize(img, resized_dimensions,
        #                             interpolation=cv2.INTER_AREA)
        text=pytesseract.image_to_string(img, lang='chi_sim')
    if method == 'blur':
        img = preprocess(img)
        # width = 700
        # scaling = width/img.shape[1]
        # height = int(img.shape[0]*scaling)
        # resized_dimensions = (width, height)
        # img = cv2.resize(img, resized_dimensions,
        #                             interpolation=cv2.INTER_AREA)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        text=pytesseract.image_to_string(img, lang='chi_sim')

    return text

# get total number of frames
fname = 'fox_spirit_ep35'
path = f'./data/{fname}.mp4'
# fname = 'fox_spirit_frame7.jpg'
# extractFrames(path)
# Function to extract frames
# processing = ['none','whiteMask','biThresh','kernel','whiteMask+kernel','blur']
processing = ['whiteMask','whiteMask+kernel','blur']
resize = ['1600w']
# resize = ['none','1600w']

with open('./data/fox_spirit_ep35_ans.txt',encoding='utf-8') as f:
    raw = f.readlines()
data = []
for line in raw:
    line = line.replace("\n","")
    data.append(line)
for size in resize:
    for m in processing:
        extractFrames(path)