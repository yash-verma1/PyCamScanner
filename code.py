import cv2 as cv
import numpy as np

im = cv.imread('test.jpg')
im = cv.resize(im,(800,600))
imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
imblur = cv.GaussianBlur(imgray,(5,5),0)
ret,thresh = cv.threshold(imblur,127,255,0)
#cv.imshow('image thesh',thresh)
#cv.waitKey(0)
contours, heirarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
c = max(contours,key=cv.contourArea)

x,y,w,h = cv.boundingRect(c)

imgResult = im[y:y+h,x:x+w]

#cv.imshow('Result',cv.drawContours(im,[c],0,(0,255,0),3))

 

# result is correct
cv.imshow('Result',imgResult)
cv.waitKey(0)
