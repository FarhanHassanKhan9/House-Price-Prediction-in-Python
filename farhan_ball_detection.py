import cv2
import numpy as np
from PIL import Image

im = Image.open('geo.jpg')
width, height = im.size
height_image=0.15*height
hei=0.03*height
path="I:\Tickers_Dataset\Annotated\image_10.jpg"

input = cv2.imread(path)


gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
aa= cv2.imshow('0', gray)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
sab= cv2.imshow('3', connected)

_,contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
mask = np.zeros(bw.shape, dtype=np.uint8)

for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, i, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    if h < height_image and h > hei:
        cv2.rectangle(input, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

cv2.imshow('output', input)
k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
