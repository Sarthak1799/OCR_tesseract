# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:09:05 2020

@author: sarthak
"""
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np

# Specify the path to installed tesseract application
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe" 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())




# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

if args["preprocess"] == 'thresh':
    final = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif args["preprocess"] == 'blur':
    final = cv2.medianBlur(gray, 3)
elif args["preprocess"] == 'deskew':
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)    
    final = rotated
elif args["preprocess"] == 'dilate':
    kernel = np.ones((5,5),np.uint8)
    final = cv2.dilate(gray,kernel,iterations=1)
elif args["preprocess"] == 'erode':
    kernel = np.ones((5,5),np.uint8)
    final = cv2.erode(gray, kernel, iterations = 1)




# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, final)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", final)
cv2.waitKey(0)

# Save the text in a textfile
with open('Parsed.txt','w') as f:
    f.writelines(text)

# Provide the input from CLI in format as follows - 
# python ocr.py -i image_name.extension -p preprocessing_option (default is thresh)