# Importing libraries
import cv2 as cv
import numpy as np
import os

def mapp(h):
	h = h.reshape((4,2))
	hnew = np.zeros((4,2),dtype = np.float32)

	add = h.sum(1)
	hnew[0] = h[np.argmin(add)]
	hnew[2] = h[np.argmax(add)]

	diff = np.diff(h,axis = 1)
	hnew[1] = h[np.argmin(diff)]
	hnew[2] = h[np.argmax(diff)]

	return hnew

img = cv.imread('test.jpg')
img = cv.resize(img,(1300,800))

#cv.imshow('image',img)
#cv.waitKey(0)

grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('Gray Scale',grayImg)
#cv.waitKey(0)

#blurredImg = cv.GaussianBlur(grayImg,(9,9),0)
blurredImg = cv.GaussianBlur(grayImg,(5,5),0)
#cv.imshow('Blurred',blurredImg)
#cv.waitKey(0)

edged = cv.Canny(blurredImg,30,50)
#cv.imshow('Canny',edged)
#cv.waitKey(0)

contours, hierarchy = cv.findContours(edged,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
"""
contours = sorted(contours,key=cv.contourArea,reverse=True)
for c in contours:
	p = cv.arcLength(c,True)
	approx = cv.approxPolyDP(c,0.02*p,True)

	if len(approx)==4:
		target=approx
		break

approx = mapp(target)
pts = np.float32([[0,0],[800,0],[800,800],[800,0]])
"""
c = max(contours,key=cv.contourArea)
op = cv.getPerspectiveTransform(approx,[c])
dst = cv.warpPerspective(img,op,(800,800))

cv.imshow("scanned",dst)
cv.waitKey(0)
