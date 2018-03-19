import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
# # # d = 25
# # in_img = cv2.imread('Input/Brute.PNG',0)
# # # in_img = cv2.medianBlur(in_img,11)
# # # # in_img=cv2.bilateralFilter(in_img,d,d,d)
# # # thresh = cv2.adaptiveThreshold(in_img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# # # cv2.imshow("thresh",thresh)
# # # ret,thresh_2 = cv2.threshold(in_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # # cv2.imshow("thresh_2",thresh_2)
# # # #
# # # cv2.imshow("gaussian",in_img)
# # # edges = cv2.Canny(in_img,25,15)
# # # # edges =
# # # cv2.imshow("edges",edges)
# # # cv2.waitKey()
# # # def blur(x):
# # #     pass
# # # def nothing(x):
# # #     pass
# #
# # Create a black image, a window
# in_img = cv2.imread('Input/Buckling_1_7077.bmp',0)
# h = 560
# w = 400
# x0 = 0
# y0 = 0
# in_img = in_img[y0:h+y0,x0:x0+w]
# # in_img = cv2.resize(in_img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)
# out_img = np.copy(in_img)
# mesh_block_size = (h/10.,w/10.)
# # for y in
# cv2.namedWindow('image')
# blur_slider=1
# a = 3
# b = 3
# s = 0
# e = 1
# f = 1
# def nothing(x):
#     pass
# def ontrackbar(blur,a,b,s,out_img):
#     if s!=0:
#         if blur %2 ==0:
#             blur +=1
#         out_img = cv2.medianBlur(out_img,blur)
#         edges = cv2.Canny(out_img,a,b)
#         out_img = edges
#         return edges
# def ontrackbar_2(blur,a,b,s,out_img):
#     if s!=0:
#         blur_1 = blur/2
#         if blur_1 % 2 ==0:
#             blur_1 +=1
#
#         if blur %2 ==0:
#             blur +=1
#
#         out_img = cv2.GaussianBlur(out_img,(blur,blur),0)
#         edges = cv2.Canny(out_img,a,b)
#         out_img = edges
#         return edges
# def ontrackbar_3(blur,a,b,s,e,f,out_img):
#     if s!=0:
#         if blur %2 ==0:
#             blur +=1
#         if f==0:
#             f=1
#         # clahe = cv2.createCLAHE(clipLimit=e, tileGridSize=(f,f))
#         # out_img = clahe.apply(out_img)
#         out_img = cv2.medianBlur(out_img,blur)
#         out_img = cv2.GaussianBlur(out_img,(blur,blur),0)
#         edges = cv2.Canny(out_img,a,b)
#         out_img = edges
#         return edges
#
# # create trackbars for color change
# cv2.createTrackbar('median_blur','image',blur_slider,100,nothing)
# cv2.createTrackbar('canny_a','image',a,1000,nothing)
# cv2.createTrackbar('canny_b','image',b,1000,nothing)
# cv2.createTrackbar('clipLimit','image',e,30,nothing)
# cv2.createTrackbar('tileGridSize','image',f,100,nothing)
# # create switch for ON/OFF functionality
# switch = '0 : OFF \n1 : ON'
# cv2.createTrackbar(switch, 'image',s,1,nothing)
#
# while(1):
#     cv2.imshow('image',out_img)
#     k = cv2.waitKey(1) & 0xFF
#
#     if k == 27:
#         break
#     blur_slider = cv2.getTrackbarPos('median_blur','image')
#     a =  cv2.getTrackbarPos('canny_a','image')
#     b =  cv2.getTrackbarPos('canny_b','image')
#     s =  cv2.getTrackbarPos(switch,'image')
#     e =  cv2.getTrackbarPos('clipLimit','image')
#     f =  cv2.getTrackbarPos('tileGridSize','image')
#     out_img = ontrackbar_3(blur_slider,a,b,s,e,f,in_img)
#     if out_img is None:
#         out_img = np.copy(in_img)
# cv2.destroyAllWindows()
#
# # Callback Function for Trackbar (but do not any work)
#
#
def column(matrix, i):
    return [row[i] for row in matrix]
filename = "Input/v_h_.txt"
results = np.genfromtxt(filename,dtype=None,delimiter="")
results = results[1:]
t = np.array(column(results,0),dtype=np.float)
D = np.array(column(results,1),dtype=np.float)
D_tild = savgol_filter(D,81,3)
D_tild_tild = savgol_filter(D_tild,21,3)

D_tild_prime = savgol_filter(D,11,3,deriv=1)

plt.plot(t,D,'r--',label='raw')
plt.plot(t,D_tild,'g*',label='raw')
plt.plot(t,D_tild_tild,'b-',label='smoothed')
plt.legend()
plt.show()