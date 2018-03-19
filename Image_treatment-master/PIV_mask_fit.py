import numpy as np
import cv2
from scipy.signal import savgol_filter
BLUR=3
CANNY = 12
THRESH_DENSITY = 5
GAUSSIANTHRESH = 3
# STEP = 449
im = cv2.imread('Input/Buckling_zoom_FPS_10k_00001.tif',0)
h,w = im.shape
x0,y0,h,w = 0,0,480,900
im_a = im[y0:y0+h,x0:x0+w]
# im_b = im[y0+h:-1,x0:x0+w]
cv2.imshow("Zone A", im_a)
# cv2.imshow("Zone B", im_b)
cv2.waitKey()
MIN_NEIGHBORS = 15

def neighboring(point,main_array,blocksize):


    if (point[0]+blocksize)<=np.amax(main_array[:,0]):
        x_max = point[0]+blocksize
    else:
        x_max = np.amax(main_array[:,0])
    if (point[0]-blocksize)>=0:
        x_min = point[0]-blocksize
    else:
        x_min = 0
    if (point[1]+blocksize)<=np.amax(main_array[:,1]):
        y_max = point[1]+blocksize
    else:
        y_max = np.amax(main_array[:,1])
    if (point[1]-blocksize)>=0:
        y_min = point[1]-blocksize
    else:
        y_min = 0

    x_centered = np.argwhere(abs(main_array[:,0]-point[0])/(0.5*(x_max-x_min))<1.0)
    y_centered = np.argwhere(abs(main_array[:,1]-point[1])/(0.5*(y_max-y_min))<1.0)
    ind = np.intersect1d(x_centered,y_centered)


    return ind.size


cv2.GaussianBlur(im,(BLUR,BLUR),0)

output_img_a = np.zeros(im_a.shape,dtype=im.dtype)
ret,thresh = cv2.threshold(im_a,150,255,0)
thresh_2 = cv2.adaptiveThreshold(im_a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, GAUSSIANTHRESH, 3)
edges_a = cv2.Canny(im_a,CANNY ,CANNY)
edges_a = thresh_2
cv2.imshow('test',edges_a)
cv2.waitKey()

index_a = np.array(np.nonzero(edges_a),dtype=np.int32)
at = index_a[0].copy()
index_a[0] = index_a[1]
index_a[1] = at
index_a = index_a.transpose()
list_y_a = []
for i in xrange(np.amin(index_a[0:,1]),np.amax(index_a[0:,1])):
    yi = np.where(index_a[:,1]==i)

    x_positions = index_a[yi]
    # max_density_index = np.argmax(densities)
    # x_considered = x_positions[max_density_index]
    try:
        yi_xmax = np.argmax(x_positions[0:,0])
        while True:
            yi_xmax = np.argmax(x_positions[0:,0])
            density = neighboring(x_positions[yi_xmax],index_a,5)
            print x_positions[yi_xmax],density
            if density > MIN_NEIGHBORS:
                break
            else:
                x_positions = x_positions[:-1,:]

        x_positions[yi_xmax][0]+=35

        list_y_a.append(x_positions[yi_xmax])
    except ValueError:
        pass
edges_a = cv2.cvtColor(edges_a,cv2.COLOR_GRAY2RGB)
array_y_a = np.array(list_y_a)
x_axis = array_y_a[:,0].copy()
array_y_a[:,0] = array_y_a[:,1]
y_axis = array_y_a[:,0]
array_y_a[:,1] = x_axis
array_y_a[:,1] = savgol_filter(array_y_a[:,1], 101, 3)
for i in xrange(1,array_y_a.shape[0]):
    pti_1=array_y_a[i-1][1],array_y_a[i-1][0]
    pti=array_y_a[i][1],array_y_a[i][0]
    cv2.line(edges_a,pti_1,pti,(0,255,0),1)
cv2.imshow('test',edges_a)
x_axis = array_y_a[:,0].copy()
array_y_a[:,0] = array_y_a[:,1]
y_axis = array_y_a[:,0]
array_y_a[:,1] = x_axis
for i in xrange(1,array_y_a.shape[0]):

    edges_a[array_y_a[i][1],array_y_a[i][0]:]=(255,255,255)
cv2.imshow('test',edges_a)
cv2.waitKey()


# output_img_b = np.zeros(im_b.shape,dtype=im.dtype)
# ret,thresh = cv2.threshold(im_b,150,255,0)
# thresh_2_b = cv2.adaptiveThreshold(im_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, GAUSSIANTHRESH, 3)
# edges_b = cv2.Canny(im_b,CANNY ,CANNY)
# # edges_b = thresh_2_b
# cv2.imshow('test',edges_b)
# cv2.waitKey()
#
# index_b = np.array(np.nonzero(edges_b),dtype=np.int32)
# at = index_b[0].copy()
# index_b[0] = index_b[1]
# index_b[1] = at
# index = index_b.transpose()
# list_y_b = []
# for i in xrange(np.amin(index[0:,1]),np.amax(index[0:,1])):
#     yi = np.where(index[:,1]==i)
#
#     x_positions = index[yi]
#     # max_density_index = np.argmax(densities)
#     # x_considered = x_positions[max_density_index]
#     try:
#         yi_xmax = np.argmax(x_positions[0:,0])
#         while True:
#             yi_xmax = np.argmax(x_positions[0:,0])
#             density = neighboring(x_positions[yi_xmax],index,5)
#             print x_positions[yi_xmax],density
#             if density > MIN_NEIGHBORS:
#                 break
#             else:
#                 x_positions = x_positions[:-1,:]
#
#         x_positions[yi_xmax][0]+=0
#
#         list_y_b.append(x_positions[yi_xmax])
#     except ValueError:
#         pass
# edges_b = cv2.cvtColor(edges_b,cv2.COLOR_GRAY2RGB)
# array_y_b = np.array(list_y_b)
# x_axis = array_y_b[:,0].copy()
# array_y_b[:,0] = array_y_b[:,1]
# y_axis = array_y_b[:,0]
# array_y_b[:,1] = x_axis
# array_y_b[:,1] = savgol_filter(array_y_b[:,1], 101, 9)
# for i in xrange(1,array_y_b.shape[0]):
#     pti_1=array_y_b[i-1][1],array_y_b[i-1][0]
#     pti=array_y_b[i][1],array_y_b[i][0]
#     cv2.line(edges_b,pti_1,pti,(0,255,0),1)
# cv2.imshow('test',edges_b)
# cv2.waitKey()