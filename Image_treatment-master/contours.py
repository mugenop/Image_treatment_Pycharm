

'''
This program illustrates the use of findContours and drawContours.
The original image is put up along with the image of drawn contours.

Usage:
    contours.py
A trackbar is put up which controls the contour level from -3 to 3
'''

import numpy as np
import cv2
import timeit
from matplotlib import pyplot as plt
from math import atan, sin
import lmfit
def number_of_points (y):
    work_array = np.where (ventouse_shape[:,1] == y )
    return ventouse_shape[work_array].shape[0]
def maxD_finder(array2d):
    xmax, ymax = np.argmax(array2d,axis=0)
    xmin, ymin = np.argmin(array2d,axis=0)
    xmin_points = array2d[np.argwhere(array2d[:,0]==array2d[xmin][0])]
    xmax_points = array2d[np.argwhere(array2d[:,0]==array2d[xmax][0])]
    xmax_points_xmax,xmax_points_ymax = np.ravel(np.amax(xmax_points,axis = 0))
    xmin_points_xmax, xmin_points_ymax = np.ravel(np.amax(xmin_points,axis = 0))
    y_max = max(xmax_points_ymax,xmin_points_ymax)
    y_max_index = np.amax(np.argwhere(array2d[:,1]==y_max))
    maxD = xmax_points_xmax-xmin_points_xmax
    return maxD,y_max,y_max_index
def resid(p):
    v = p.valuesdict()
    experimental_R = np.sqrt((ball_shape[:,0]- v['x0'])**2+ (ball_shape[:,1]-v['y0'])**2)
    theta = np.arctan2((ball_shape[:,1]-v['y0']),( ball_shape[:,0]-v['x0']))
    theoretical_R  = 1
    for i in xrange (1,PARAMETER_NUMBER+1):
        name = 'a'+str(i)
        theoretical_R  += v[name]*(np.sin(theta))**i
    theoretical_R  = v['b0']* theoretical_R
    return (experimental_R-theoretical_R)

def circular(p):
    v = p.valuesdict()
    experimental_R = np.sqrt((circle_ball_shape[:,0]- v['x0'])**2+ (circle_ball_shape[:,1]-v['y0'])**2)
    theta = np.arctan2((circle_ball_shape[:,1]-v['y0']),( circle_ball_shape[:,0]-v['x0']))
    theoretical_R = v['b0']*(1+v['a1']*np.sin(theta))
    return (experimental_R-theoretical_R)

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

BALLTHRESHOLD = 195
VENTOUSETHRESHOLD = 70
PARAMETER_NUMBER = 9
if __name__ == '__main__':
    print __doc__

    # img = np.zeros((500, 500), np.uint8)
    # print type(img)
    img = cv2.imread('RedBall_registered00000.bmp',0)
    img = img [640:2176,212:724]
    # img = img[70:2501,180:875]
    print img.shape
    h,  w= img.shape[:2]
    vis = np.zeros((h, w, 3), np.uint8)
    # img = cv2.bitwise_not(img)
    # dstimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t0= timeit.default_timer()
    img = cv2.medianBlur(img,51)
    t01 = timeit.default_timer() - t0
    print 'time to blur = ' +str(t01)

    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,2)
    # thresh = rotateImage(thresh,)

    t1 = timeit.default_timer() -  t0
    print 'time to thresh = ' +str(t1)
    contours0, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t2 = timeit.default_timer() - t0 - t1
    print 'time to get contours = ' +str(t2)
    contours01 = np.array(contours0)
    final_contours = np.ravel(contours01[0])
    for i in xrange (1, contours01.shape[0]):
        intermediate = np.ravel(contours01[i])
        final_contours = np.concatenate((final_contours,intermediate))

    final_contours = np.reshape(final_contours,(final_contours.shape[0]/2,2))
    t3 = timeit.default_timer() -t2 - t1 - t0
    print "time to create matrix = "+ str(t3)
    final_contours = np.unique(final_contours.view(np.dtype((np.void, final_contours.dtype.itemsize*final_contours.shape[1])))).view(final_contours.dtype).reshape(-1, final_contours.shape[1])
    # print ys
    # t30 = timeit.default_timer() -t2 - t1 - t0 -t3
    # print "time to erase redundancy = "+str(t30)

    ind = np.lexsort((final_contours[:,0],final_contours[:,1]))
    final_contours = final_contours[ind]
    t4 = timeit.default_timer()-t3-t2-t1-t0
    general_time = timeit.default_timer()-t0
    print "time to sort = "+str(t4)
    # ys = set(final_contours[:,1])
    # ds = map (D_finder,ys)
    maxD,y_max,y_max_index0 = maxD_finder(final_contours)
    y_max_index = y_max_index0+1
    maxDiameter = maxD
    while maxD > BALLTHRESHOLD:
        maxD, y_max,dy_max_index = maxD_finder(final_contours[y_max_index:-1])
        y_max_index +=dy_max_index+1
    ball_shape = final_contours[:y_max_index]
    ball_shape_index = y_max_index
    t5 =  timeit.default_timer()-t0-t1-t2-t3-t4
    print "time to find ball shape = "+str (t5)
    work_array = np.where(final_contours[:,1] == 79 )
    maxD, y_max,y_max_index0 = maxD_finder(final_contours[ball_shape_index:])
    y_max_index =ball_shape_index + y_max_index0+1
    while maxD > VENTOUSETHRESHOLD:
        maxD, y_max,dy_max_index = maxD_finder(final_contours[y_max_index:-1])
        y_max_index += dy_max_index+1
    ventouse_shape_index = y_max_index
    ventouse_shape = final_contours[ball_shape_index:ventouse_shape_index]
    t6 =  timeit.default_timer()-t0-t1-t2-t3-t4-t5
    print "time to find ventouse shape = "+str (t6)
    vis2 = np.zeros((h, w, 3), np.uint8)

    for i in xrange(ventouse_shape.shape[0]):
        number_of_points_array = map(number_of_points,ys_ventouse)
        y_number_of_points_max_index = np.argmax(number_of_points_array)
        y_number_of_points_max = np.array(ys_ventouse)[y_number_of_points_max_index]
        y_ventouse_bottom = np.amax(ventouse_shape[:,1],axis=0)
        line = ventouse_shape[np.where(ventouse_shape[:,1]==np.amax(ventouse_shape[:,1],axis=0))]
        cv2.line(vis2,(line[0][0],line[0][1]),(line[-1][0],line[-1][1]),(0,0,204),3)
        maxD,y_max,y_max_index0 = maxD_finder(ball_shape)
        y_max_index = y_max_index0+1
    while maxD > 0.95*maxDiameter:
        maxD, y_max,dy_max_index = maxD_finder(ball_shape[y_max_index:-1])
        y_max_index += dy_max_index+1
    y_stitcher= y_max
    circle_ball_shape = ball_shape[y_max_index:]
    x0array = np.mean(ball_shape,0)
    pcircular = lmfit.Parameters()
    pcircular.add('a1', value = 0.0)#negatif
    pcircular.add('b0', value =maxDiameter/2.0 ,min = 0.0, )
    pcircular.add('x0',value = x0array[0])
    pcircular.add('y0',value = x0array[1])
    mi_circular = lmfit.minimize(circular,pcircular)
    lmfit.printfuncs.report_fit(pcircular)
    theta = np.arange(0, 2*np.pi, np.pi/360)
    cirle_part = pcircular['b0']*(1+pcircular['a1']*np.sin(theta))
    Xthc = (pcircular['x0']+cirle_part* np.cos(theta)).astype(int)
    Ythc = (pcircular['y0']+cirle_part * np.sin(theta)).astype(int)
    circle_ball_array = np.ndarray((Xthc.shape[0],2))
    circle_ball_array[:,0] = Xthc
    circle_ball_array[:,1] = Ythc
    ind = np.lexsort((circle_ball_array[:,0],circle_ball_array[:,1]))
    circle_ball_array = circle_ball_array[ind]
    y_stitcher = np.amax(ball_shape[:,1],axis=0)
    stitching_index = np.argwhere(circle_ball_array[:,1]>=y_stitcher)
    while stitching_index.shape[0]<100:
        y_stitcher = np.amax(ball_shape[:,1],axis=0)
        y_stitcher_index =  np.argwhere(ball_shape[:,1]==y_stitcher)
        ball_shape=np.delete(ball_shape,y_stitcher_index,0)
        stitching_index = np.argwhere(circle_ball_array[:,1]>=y_stitcher)

    circle_ball_array = (circle_ball_array[stitching_index[0]:]).astype(int)



    ball_shape = np.concatenate((ball_shape,circle_ball_array),axis=0)
    p9order = lmfit.Parameters()
    for i in xrange(1,PARAMETER_NUMBER+1):
        name = 'a'+str(i)
        p9order.add(name, value = 0.0)
    p9order.add('b0', value =maxDiameter/2.0 ,min = 0.0, )
    p9order.add('x0',value = x0array[0])
    p9order.add('y0',value = x0array[1])
    mi = lmfit.minimize(resid, p9order)
    v =p9order.valuesdict()
    theta = np.arange(0,2*np.pi,np.pi/180)
    Fitted_contour = 1
    for i in xrange (1,PARAMETER_NUMBER+1):
        name = 'a'+str(i)
        Fitted_contour += v[name]*(np.sin(theta))**i
    Fitted_contour = v['b0']* Fitted_contour
    # Fitted_contour = v['b0']*(1+v['a1']*np.sin(theta))
    Xth = v['x0']+Fitted_contour* np.cos(theta)
    Yth = v['y0']+Fitted_contour * np.sin(theta)
    Xth = Xth.astype(int)
    Yth = Yth.astype(int)
    y_bas_index = np.argmax(Yth,axis=0)
    y_bas = Yth[y_bas_index]
    x_bas = Xth[y_bas_index]
    for i in xrange(Xth.shape[0]):
        cv2.line(vis2,(Xth[i],Yth[i]),(Xth[i],Yth[i]),(0,255,0),3)
    # vis [v['y0']][v['x0']] = 255
    # img[Xth][Yth]= (255,255,255)

    lmfit.printfuncs.report_fit(p9order)
    print mi.success
    print mi.nfev
    print mi.residual[-1]
    # x = np.arange(0, final_contours.shape[0],1)
    # y = final_contours[:,1]
    # plt.plot(x,y)
    # plt.show()
    # print final_contours[xmin]
    # print np.amax(final_contours,axis=0)
    # # for i in xrange(N.shape[0]):
    # #     print N[i]
    #############################################################################################

    # # cv2.namedWindow('contours2',cv2.cv.CV_WINDOW_NORMAL)
    # # cv2.namedWindow('contours',cv2.cv.CV_WINDOW_NORMAL)
    # cv2.drawContours( img,contours0, (-1, 3)[levels <= 0], (128,255,255), 1, cv2.CV_AA, None, abs(levels) )
    # cv2.line(img,(line[0][0],y_number_of_points_max),(line[-1][0],y_number_of_points_max),(201,0,0),10)
    cv2.line(vis2,(p9order['x0'],p9order['y0']),(p9order['x0'],p9order['y0']),(255,255,255),10)
    cv2.line(vis2,(x_bas,y_bas),(x_bas,y_bas),(255,255,255),10)
    # # # # print np.argmax(N,axis=1)
    cv2.imshow('image',vis2)
    cv2.imwrite('RedBall720_filled.bmp',vis2)

    # # cv2.imshow('contours', vis)
    # # cv2.imshow('contours2', vis2)
    cv2.waitKey()
    # # # # for i in xrange(len(contours0)):
    # # # #         N += contours0[i]
    # # # t3 = timeit.default_timer() - t2 - t1 -t0
    # # # print 'time to concatenate = ' +str(t3)
    # # # N = np.concatenate(contours0)
    # # # alpha = cv2.boundingRect(N)
    # # # print alpha
    # # # alpha2 = [alpha[i:i+2] for i in range(0, len(alpha), 2)]
    # # #
    # # #
    # # # vis2 = np.zeros((h, w, 3), np.uint8)

    # # # cv2.namedWindow('image',cv2.cv.CV_WINDOW_NORMAL)
    # # # cv2.namedWindow('image',cv2.cv.CV_WINDOW_NORMAL)
    # # #
    # # # levels = 1
    # # # #
    # # # cv2.namedWindow('contours2',cv2.cv.CV_WINDOW_NORMAL)
    # # # cv2.drawContours( vis2, contours0, (-1, 3)[levels <= 0], (128,255,255), 1, cv2.CV_AA, None, abs(levels) )
    # # # cv2.rectangle(vis2,(alpha[0],alpha[1]),(alpha[0]+alpha[2],alpha[1]+alpha[3]),(255,0,0),1)
    # # # cv2.imshow('image',thresh)
    # # #
    # # # cv2.imshow('contours2', vis2)
    # # # cv2.waitKey()
    # # # # for i in xrange(len(contours0)/2):
    # # # #     cv2.drawContours( vis, contours0[i], (-1, 3)[levels <= 0], (128,255,255), 2, cv2.CV_AA, None, abs(levels) )
    # # # #     cv2.waitKey(1)
    # # # #
    # cv2.imwrite('RedBall0_skeleton.bmp',vis)
    # # #
    # # # # # vis = cv2.medianBlur(vis,7)
    # #
