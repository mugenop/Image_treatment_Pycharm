import numpy as np
import cv2
from scipy.signal import savgol_filter
BLUR=3
CANNY = 30
THRESH_DENSITY = 5
GAUSSIANTHRESH = 5
STEP = 449
im = cv2.imread('Input/Buckling_zoom_FPS_10k_00001.tif',0)
# x0,y0,h,w = 900,0,450,480
# im = im[y0:y0+h,x0:x0+w]
MIN_NEIGHBORS = 5

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

output_img = np.zeros(im.shape,dtype=im.dtype)
ret,thresh = cv2.threshold(im,150,255,0)
thresh_2 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, GAUSSIANTHRESH, 3)
edges = cv2.Canny(im,CANNY ,CANNY)
# edges = thresh_2
a = np.where(edges[np.nonzero(edges)]<255)[0].size
cv2.imshow('test',edges)
cv2.waitKey()
# index = np.array(np.nonzero(edges),dtype=np.int32)
# at = index[0].copy()
# index[0] = index[1]
# index[1] = at
# index = index.transpose()
# list_y = []
# for i in xrange(np.amin(index[0:,1]),np.amax(index[0:,1])):
#     yi = np.where(index[:,1]==i)
#
#     x_positions = index[yi]
#     # max_density_index = np.argmax(densities)
#     # x_considered = x_positions[max_density_index]
#     try:
#         yi_xmax = np.argmax(x_positions[0:,0])
#         # while True:
#         #     yi_xmax = np.argmax(x_positions[0:,0])
#         #     density = neighboring(x_positions[yi_xmax],index,5)
#         #     print x_positions[yi_xmax],density
#         #     if density > MIN_NEIGHBORS:
#         #         break
#         #     else:
#         #         x_positions = x_positions[:-1,:]
#
#         # x_positions[yi_xmax][0]+=15
#         print x_positions[yi_xmax]
#         list_y.append(x_positions[yi_xmax])
#     except ValueError:
#         pass
# edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
# try:
#     edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
# except cv2.error:
#     pass
# array_y = np.array(list_y)
# x_axis = array_y[:,0].copy()
# array_y[:,0] = array_y[:,1]
# y_axis = array_y[:,0]
# array_y[:,1] = x_axis
# array_y[:,1] = savgol_filter(array_y[:,1], 11, 1)
# for i in xrange(1,array_y.shape[0]):
#     pti_1=array_y[i-1][1],array_y[i-1][0]
#     pti=array_y[i][1],array_y[i][0]
#     cv2.line(edges,pti_1,pti,(0,255,0),1)
#
#
# cv2.imshow('test',edges)
# cv2.waitKey()
# index = STEP
# y_max_kp = 0
#
#
# xmin = np.amin(x_axis)
# xmax = np.amax(x_axis)
# bins = np.arange(0,xmax+1)
# x_average = np.bincount(x_axis)
# print x_average[0]
# print array_y[:,1]
# x_average_array = bins[np.argwhere(x_average>THRESH_DENSITY)]
# x_average_y= []
# points_list= []
# x_indexes = np.argwhere(x_axis==x_average_array)
# points =  array_y[x_indexes].flatten()
# points = points.reshape((points.shape[0]/2, 2))
# ind = np.lexsort((points[:,1],points[:,0]))
# points = points[ind]
# cpoints = points.view('c8')
# cpoints = np.unique(cpoints)
# points = cpoints.view('i4').reshape((-1,2))
#
#
# k = 10
# for i in xrange(k,points.shape[0],k):
#     pti_1=points[i-k][1],points[i-k][0]
#     pti=points[i][1],points[i][0]
#     cv2.line(edges,pti_1,pti,255,1)
#     cv2.line(output_img,pti_1,pti,255)
#
# pti_1=points[i][1],points[i][0]
# pti=points[-1][1],points[-1][0]
# cv2.line(edges,pti_1,pti,(255,0,0),1)
# cv2.line(output_img,pti_1,pti,255)
# pti_1=points[-1][1],points[-1][0]
# pti=w,points[-1][0]
# cv2.line(edges,pti_1,pti,(255,0,0),1)
# cv2.line(output_img,pti_1,pti,255)
# cv2.imshow('test',edges)
# cv2.waitKey()
#
# index = np.array(np.nonzero(output_img),dtype=np.int32)
# at = index[0].copy()
# index[0] = index[1]
# index[1] = at
# index = index.transpose()
# index = array_y
# for i in xrange(index.shape[0]):
#     Y = index[i][1]
#     X = index[i][0]
#     im[Y,X:]=255
# cv2.imshow('test',im)
# cv2.waitKey()
# for i in xrange(x_average_array.shape[0]):
#     points =  array_y[np.where(x_axis==x_average_array[i])]
#     x_average_y.append(np.amax(points[:,0]))
# try:
#     ymax = np.amax(np.array(x_average_y))
#     ymax_index = np.argmax(np.array(x_average_y))
#     xmax = x_average_array[ymax_index]
#     print xmax,ymax
#
#     im[:ymax,xmax:] = 0
# except ValueError:
#     print x_average_y
#
#
# for i in xrange(array_y.shape[0]):
#     output_img[array_y[i][0],array_y[i][1]]=255
#
#
