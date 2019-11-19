import cv2 as cv
import numpy as np

im = cv.imread('test.jpg')
im = cv.resize(im,(800,600))
imgray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray,127,255,0)
contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

print(len(contours))

c = max(contours,key = cv.contourArea)
cv.imshow('img',cv.drawContours(im,[c],0,(0,255,0),3))
cv.waitKey(0)
