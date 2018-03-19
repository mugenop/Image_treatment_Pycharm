__author__ = 'Adel'
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
# x0 =197
# y0 =483
# h  = 566-y0
# w  = 414-x0
# a = 100
# x0 =0
# y0 =0
# h  = 300-y0
# w  = 310-x0
# a = 5
# b = 51
# c = 11
# d = 13
# e = 7
# FILE_IN = 'ROI'
# FILE_OUT = FILE_IN+'_'
# EXTENSION = '.bmp'
# in_img = cv2.imread('Input/'+FILE_IN+EXTENSION,0)
# h,w= in_img.shape[:2]
#
# img = in_img[y0:y0+h,x0:x0+w]
# # for i in xrange (0,100):
# #     print i
# #     blur = cv2.bilateralFilter(img,i,3,3)
# #     cv2.imwrite('Output/Blur/'+FILE_OUT+str(i)+EXTENSION,blur)
# #     cv2.waitKey()
# blur = cv2.GaussianBlur(img,(a,a),0)
#
# blur2 = cv2.medianBlur(img,e)
# blur3 = cv2.bilateralFilter(img,d,c,c)
# edges = cv2.Canny(blur,50,20)
# cv2.imshow("canny1",edges)
# index = np.nonzero(edges)
# img1 = img.copy()
# img1[index]= 255
# # img1 = cv2.bitwise_xor(img,img,mask = edges)
# cv2.imshow("contour1",img1)
# edges = cv2.Canny(blur2,50,20)
# cv2.imshow("canny2",edges)
# img2 = img + edges
#
# cv2.imshow("contour2",img2)
# edges = cv2.Canny(blur3,50,20)
# cv2.imshow("canny3",edges)
# img3 = img + edges
#
# cv2.imshow("contour3",img3)
# cv2.waitKey()

# # plt.hist(img.ravel(),256,[0,256]); plt.show()
# blur0 = cv2.equalizeHist(blur)
# cv2.imwrite('Output/Hist/'+FILE_OUT+EXTENSION,blur0)
#
# # plt.hist(blur0.ravel(),256,[0,256]); plt.show()
#
# # blur0 = blur
# # cv2.imshow("CONTRAST",blur0)
# # for i in xrange (3,100):
# #     if i%2 ==1:
# #         print i
# #         blur = cv2.medianBlur(blur0,i)
# #         cv2.imwrite('Output/Blur/'+FILE_OUT+str(i)+EXTENSION,blur)
# blur2 = cv2.medianBlur(blur0,11)
# # cv2.imshow("Blur2",blur)
#
# ret3,th3 = cv2.threshold(blur2,127,255,cv2.THRESH_BINARY)
# cv2.imwrite('Output/Threshold/'+FILE_OUT+EXTENSION,th3)
# # for i in xrange (3,100):
# #     if i%2 ==1:
# #         print i
# #         img0 = np.copy(img)
# #         th3 = cv2.adaptiveThreshold(blur0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,i,2)
# #         cv2.imwrite('Output/Blur/'+FILE_OUT+str(i)+EXTENSION,th3)
# #         contours,hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# #         cnt = contours[0]
# #         cv2.drawContours(img0, [cnt], -1, (255,255,255), 1)
# #         cv2.imwrite('Output/Contour/'+FILE_OUT+str(i)+EXTENSION,img0)
# contours,hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# cnt = contours[0]
# cv2.drawContours(img, [cnt], -1, (255,255,255), 1)
# cv2.imwrite('Output/Contour/'+FILE_OUT+str(0)+EXTENSION,img)


# for i in xrange (3,255):
#     if i%1 ==0:
#         img0 = np.copy(img)
#         blur = blur0
#         print i
#         ret3,th3 = cv2.threshold(blur,i,255,cv2.THRESH_BINARY)
#         cv2.imwrite('Output/Threshold/'+FILE_OUT+str(i)+EXTENSION,th3)
#         contours,hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#         cnt = contours[0]
#         cv2.drawContours(img0, [cnt], -1, (0,255,0), 1)
#         cv2.imwrite('Output/Contour/'+FILE_OUT+str(i)+EXTENSION,img0)
#         # # cv2.imshow("thresh",th3)
# contours,hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# cnt = contours[0]
# cv2.drawContours(img, [cnt], -1, (0,255,0), 1)
# cv2.imwrite('Output/'+FILE_OUT+str(1)+EXTENSION,img)


# for i in xrange (80,255):
#     if i%1==0:
#         print i
#         in_img = cv2.imread('Input/Test3207.jpg',0)
#         img = in_img[y0:y0+h,x0:x0+w]
#         blur = cv2.bilateralFilter(img,93,5,5)
#         blur = cv2.equalizeHist(blur)
#         blur = cv2.medianBlur(img,51)
#         ret3,th3 = cv2.threshold(blur,i,255,cv2.THRESH_BINARY_INV)
#         contours,hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#         cnt = contours[0]
#         cv2.drawContours(img, [cnt], -1, (0,255,0), 1)
#         cv2.imwrite('Output/'+FILE_OUT+str(i)+'.jpg',img)

# Contour detection and drawing
# ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# # Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# # plot all the images and their histograms
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
#
# for i in xrange(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()
#




# Smoothing: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
# Threasholding : http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
# Contour: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
def list_remover(list):
    list.append('bite')
str = '-10'
print int(str)
time = np.arange(0.0,0.01,0.001)
print time.shape

final_array = np.ones((25),dtype=bool)
print final_array
general_path_in =   "D:\\Documents\\Footage\\videos\\Manip_finales\\Water\\Ressort\\5_25\\Quasi_static\\Manip_1\\Input\\"
general_path_out =  "D:\\Documents\\Footage\\videos\\Manip_finales\\Water\\Ressort\\5_25\\Quasi_static\\Manip_1\\Output\\"

b_isotropic_path_in = "Buckling\\Isotropic\\"

images_list = os.listdir(general_path_in+b_isotropic_path_in)
print images_list
images_list.reverse()
print images_list

list_remover(images_list)
print images_list