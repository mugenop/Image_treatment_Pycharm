__author__ = 'Adel'

import numpy as np
import cv2
import timeit
from matplotlib import pyplot as plt
from math import atan, sin
import lmfit
ITERATION_YMAX = 100
ITERATION_CONCAVITY = 2
"""TODO:
   FAIRE UNE METHODE POUR FILTRER LES CONTOURS DONT ON NE VEUT PAS DANS CONCAVITY

"""
class Shape():
    '''
    Path_in is the original image path.
    Path_out is the output path of the processed image.
    whiteOnBlack means that -in the original THRESHOLDED image- background is black and lines are white, usually it's the opposite.
    img is a constructor by copy "surcharge"
    Thresholds are a dictionnary of the different thresholds to be used in the image process.

    '''
    def __init__(self, path_in, thresholds=None, path_out=None, img=None):
        self.in_name = path_in
        self.out_name = path_out
        self.in_img = None
        self.thresholds = thresholds
        if (img is None):
            self.load()
        else:
            self.in_img = img
        if self.in_img is not None:
            self.h,  self.w= self.in_img.shape[:2]
            self.out_img = np.zeros((self.h, self.w, 3), np.uint8)
        self.general_contour = None
        # self.guassian_mean_threshold()
        # self.find_general_shape()
        # index = self.find_ball_shape()
        # self.find_ventouse_shape(index)

    def load(self):
        self.in_img = cv2.imread(self.in_name,0)

    def median_blur(self,order=1,img=None):
        if img is None:
            if order == 1:
                self.in_img = cv2.medianBlur(self.in_img,self.thresholds['medianblur1'])
            else:
                self.in_img = cv2.medianBlur(self.in_img,self.thresholds['medianblur2'])
        else:
            if order == 1:
                img = cv2.medianBlur(img,(self.thresholds['medianblur1']))
            else:
                img = cv2.medianBlur(img,(self.thresholds['medianblur2']))
            return img

    def gaussian_blur(self,order=1,img=None):
        if img is None:
            if order == 1:
                self.in_img = cv2.GaussianBlur(self.in_img,(self.thresholds['medianblur1'],self.thresholds['medianblur1']),0)
            else:
                self.in_img = cv2.GaussianBlur(self.in_img,(self.thresholds['medianblur2'],self.thresholds['medianblur2']),0)
        else:
            if order == 1:
                img = cv2.GaussianBlur(img,(self.thresholds['medianblur1'],self.thresholds['medianblur1']),0)
            else:
                img = cv2.GaussianBlur(img,(self.thresholds['medianblur2'],self.thresholds['medianblur2']),0)
            return img

    def guassian_mean_threshold(self,img=None):
        if img is None:
            self.thresh = cv2.adaptiveThreshold(self.in_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.thresholds['gaussianthresh'], 2)
        else:
            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.thresholds['gaussianthresh'], 2)
            return thresh

    def canny(self,a,b,img=None,show_flag = False):
        if img is None:
            edges = cv2.Canny(self.in_img,a,b)
            index = np.array(np.nonzero(edges),dtype=np.int32)
            # imgd = np.zeros(self.in_img.shape,dtype=self.in_img.dtype)
            # imgd[edges] = 255
            # cv2.imshow("sdflmslkff",edges)
        else:
            if show_flag:
                cv2.imshow("Img",img)
            edges = cv2.Canny(img,a,b)
            index = np.array(np.nonzero(edges),dtype=np.int32)
        if show_flag:

            imgd = np.zeros(self.in_img.shape,dtype=self.in_img.dtype)
            imgd[edges] = 255
            cv2.imshow("Canny",edges)
            cv2.waitKey(1)

        # A verifier...
        a = index[0].copy()
        index[0] = index[1]
        index[1] = a
        index = index.transpose()
        return index

    def find_general_shape(self,a,b,show_flag_canny=False):
        self.general_contour = self.canny(a,b,show_flag=show_flag_canny)

    def sort_general_shape(self):


        """Cette methode nous permet d'ordonner le tableau de coordonnees du plus petit y au plus grand"""
        k = self.general_contour.dtype.itemsize*self.general_contour.shape[1]
        a = (np.void, k)
        v = np.dtype(a)
        d = self.general_contour.dtype
        c = self.general_contour.view(v)
        b = np.unique(c).view(d)
        self.general_contour = b.reshape(-1, self.general_contour.shape[1])
        ind = np.lexsort((self.general_contour[:,0],self.general_contour[:,1]))
        self.general_contour = self.general_contour[ind]

    def find_ball_shape(self):

        """If this method is to work, one should keep threshold as low as possible"""
        try:
            maxD, y_max,y_max_index0 = self.maxD_finder(self.general_contour)
            self.maxDiameter = maxD
            y_max_index = y_max_index0+1
            while maxD > self.thresholds['ballthreshold']:
                maxD, y_max,dy_max_index = self.maxD_finder(self.general_contour[y_max_index:-1])
                y_max_index += dy_max_index+1
            self.ball_shape = self.general_contour[:y_max_index]
            return y_max_index
        except TypeError:
            pass



    def maxD_finder(self,array2d,xreturn=False):
        try:
            xmax, ymax = np.argmax(array2d,axis=0)
            xmin, ymin = np.argmin(array2d,axis=0)
            test_a = array2d[xmin][0]
            test_b = array2d[xmax][0]
            xmin_points = array2d[np.argwhere(array2d[:,0]==array2d[xmin][0])]
            xmax_points = array2d[np.argwhere(array2d[:,0]==array2d[xmax][0])]
            xmax_points_xmax,xmax_points_ymax = np.ravel(np.amax(xmax_points,axis = 0))
            xmin_points_xmax, xmin_points_ymax = np.ravel(np.amax(xmin_points,axis = 0))
            y_max = max(xmax_points_ymax,xmin_points_ymax)

            y_max_indexes = np.argwhere(array2d[:,1]==y_max)
            y_max_index = np.amax(y_max_indexes)
            maxD =  xmax_points_xmax-xmin_points_xmax

            if xreturn:
                return xmax_points_xmax,xmin_points_xmax,y_max,y_max_index
            return maxD,y_max,y_max_index
        except ValueError:
            print 'error'

    def ball_shape_histogram(self):

        y_k = np.unique(self.ball_shape[:,1])
        D_map = np.zeros(y_k.shape,y_k.dtype)
        maxD_value = self.thresholds["maxD_filter"]
        for i in range(y_k.shape[0]):
            y_k_index = np.argwhere(self.ball_shape[:,1]==y_k[i])
            max = self.ball_shape[y_k_index[-1]][0,0]
            min = self.ball_shape[y_k_index[0]][0,0]
            maxD_k = max-min
            D_map[i] = maxD_k
            if maxD_k<maxD_value:
                self.ball_shape = np.delete(self.ball_shape,y_k_index,0)
            else:
                break

    def find_concavity(self, horizontal_shrink, vertical_offset, a, b, show_flag=False, clahe = True,vertical_negative_offset = 0):

        y_min= self.ball_shape[0][1]+vertical_offset
        xmaxf,xminf,y_max,y_max_index = self.maxD_finder(self.ball_shape[:],xreturn=True)
        xmin = xminf+int((xmaxf-xminf)*horizontal_shrink)
        xmax = xmaxf-int((xmaxf-xminf)*horizontal_shrink)
        y_max -= vertical_negative_offset
        print "Y_max: "+str(y_max)
        if xmax-xmin<50:
            xmax = 50+xmin
        while y_max-y_min<= 10:
            y_max+=1
        roi_img = self.out_img[y_min:y_max,xmin:xmax].copy()
        if self.thresholds['clipLimit'] ==0:
            clahe = False
        # cv2.imshow("ROI",roi_img)
        # cv2.waitKey(1)
        if not clahe:
            roi_img= self.median_blur(order=2,img=roi_img)
            roi_img= self.gaussian_blur(order=2,img=roi_img)

        else:
            clahe = cv2.createCLAHE(clipLimit=self.thresholds['clipLimit'], tileGridSize=(self.thresholds['tileGridSize_a'],self.thresholds['tileGridSize_b']))
            roi_img = clahe.apply(roi_img)
            roi_img= self.median_blur(order=2,img=roi_img)
            roi_img= self.gaussian_blur(order=2,img=roi_img)

        edges = self.canny(a,b,roi_img,show_flag=show_flag)
        edges[:,0] += xmin
        edges[:,1] += y_min
        self.concavity_shape = edges

    def stich(self,x0):
        maxD, y_max,y_max_index0 = self.maxD_finder(self.ball_shape[:])
        y_max_index =y_max_index0+1
        while maxD > self.thresholds['circlefitthreshold']*self.maxDiameter:
            maxD, y_max,dy_max_index = self.maxD_finder(self.ball_shape[y_max_index:-1])
            y_max_index += dy_max_index+1
        y_stitcher= y_max
        self.circle_ball_shape = self.ball_shape[y_max_index:]
        pcircular = lmfit.Parameters()
        pcircular.add('a1', value = 0.0)#negatif
        pcircular.add('b0', value =self.maxDiameter/2.0 ,min = 0.0, )
        pcircular.add('x0',value = x0[0])
        pcircular.add('y0',value = x0[1])
        mi_circular = lmfit.minimize(self.circular,pcircular)
        theta = np.arange(0, 2*np.pi, np.pi/360)
        cirle_part = pcircular['b0']*(1+pcircular['a1']*np.sin(theta))
        Xthc = (pcircular['x0']+cirle_part* np.cos(theta)).astype(int)
        Ythc = (pcircular['y0']+cirle_part * np.sin(theta)).astype(int)
        circle_ball_array = np.ndarray((Xthc.shape[0],2))
        circle_ball_array[:,0] = Xthc
        circle_ball_array[:,1] = Ythc
        ind = np.lexsort((circle_ball_array[:,0],circle_ball_array[:,1]))
        circle_ball_array = circle_ball_array[ind]
        stitching_index = np.argwhere(circle_ball_array[:,1]>=y_stitcher)
        while stitching_index.shape[0]<self.thresholds['stiching']:
            y_stitcher = np.amax(ball_shape[:,1],axis=0)
            y_stitcher_index =  np.argwhere(ball_shape[:,1]==y_stitcher)
            ball_shape=np.delete(ball_shape,y_stitcher_index,0)
            stitching_index = np.argwhere(circle_ball_array[:,1]>=y_stitcher)
        circle_ball_array = (circle_ball_array[stitching_index[0]:]).astype(int)
        self.ball_shape = np.concatenate((self.ball_shape,circle_ball_array),axis=0)
        return pcircular

    def external_shape_fitter(self,x0,theta_free=False,verbose=False):
        max_external_parameters_value = self.thresholds['max_external_parameters_value']
        self.external_shape_parameters = lmfit.Parameters()
        self.external_shape_parameters.add('b0', value =1,vary = False )
        self.external_shape_parameters.add('x0',value = x0[0],max = x0[0]+max_external_parameters_value)
        self.external_shape_parameters.add('y0',value = x0[1],max = x0[1]+max_external_parameters_value)
        if theta_free is not True:
            self.external_shape_parameters.add('theta0', value=0.0,min=(0.001)*np.pi, max=(0.03)*np.pi)
        else:
            self.external_shape_parameters.add('theta0', value=0.0,vary=False)
        for i in xrange(0,self.thresholds['fit_parameters_nbr']+1):
            name = 'a'+str(i)
            self.external_shape_parameters.add(name, value = 0.0,min = -max_external_parameters_value, max = max_external_parameters_value)


        mi = lmfit.minimize(self.calculateR,self.external_shape_parameters,args=(self.ball_shape,self.thresholds['fit_parameters_nbr']),method='leastsq',maxfev = 10000)
        self.v = self.external_shape_parameters.valuesdict()
        print "success: " + str(mi.success)
        if mi.success:
            if verbose:
                verbose_dict = {}
                verbose_dict["message"] = mi.message
                verbose_dict["residual"] = mi.residual
                verbose_dict["Iteration_number"] = mi.nfev
                lmfit.printfuncs.report_fit(self.external_shape_parameters)
        return mi.success

    def concavity_fitter(self, xc, yc,tan_alpha,verbose=False):
        self.concavity_shape_parameters = lmfit.Parameters()
        self.concavity_shape_parameters.add('xc', value=xc, vary=False)
        self.concavity_shape_parameters.add('yc', value=yc, vary=False)
        self.concavity_shape_parameters.add('tan_alpha', value=tan_alpha, vary=False)
        self.concavity_shape_parameters.add('a6', value = 0,vary=True)
        self.concavity_shape_parameters.add('a4', value=1)
        self.concavity_shape_parameters.add('a0', expr ='-a2*xc**2-a4*xc**4-a6*xc**6+yc')
        self.concavity_shape_parameters.add('a2', expr ="(-2*a4*(xc**2)-3*a6*xc**4)+(tan_alpha/(2*xc))")
        mi = lmfit.minimize(self.calculate_concavity, self.concavity_shape_parameters,method='leastsq',maxfev = 100000,ftol=1e-8,xtol=1e-8)
        self.v_c = self.concavity_shape_parameters.valuesdict()
        print "success: " + str(mi.success)
        if mi.success:
            if verbose:
                verbose_dict = {}
                verbose_dict["message"] = mi.message
                verbose_dict["residual"] = mi.residual
                verbose_dict["Iteration_number"] = mi.nfev
                lmfit.printfuncs.report_fit(self.concavity_shape_parameters)
                # print mi.message
                # print mi.residual
                # print mi.nfev
        return mi.success
        "ancienne methode"
        # self.concavity_shape_parameters = lmfit.Parameters()
        # self.concavity_shape_parameters.add('xc', value=x_c,vary=False)
        # self.concavity_shape_parameters.add('yc', value=y_c,vary=False)
        # self.concavity_shape_parameters.add('a0', value=1.)
        # self.concavity_shape_parameters.add('a2',expr = '2*(yc-a0)/(xc**2)')
        # self.concavity_shape_parameters.add('a4',expr = '(a0-yc)/(xc**4)')
        # mi = lmfit.minimize(self.calculate_concavity, self.concavity_shape_parameters)

    def external_shape_fit(self,concavity_flag = True,verbose = False,multiple_angles_flag = False, circular=False):

        x0 = np.mean(self.ball_shape,0)

        if concavity_flag:
            self.stich(x0)
            success = self.external_shape_fitter(x0,verbose=verbose,theta_free=True)
        else:
            if circular:
                pc = self.stich(x0)
                success = self.external_shape_fitter(x0,theta_free=True,verbose=verbose)
            else:
                self.stich(x0)
                success = self.external_shape_fitter(x0,theta_free=True)
        return success

    def calculate_fit_angle(self,analytical=False):
        tangent_theta = self.tangent(method_polynomial=analytical)
        max_theta = self.R_maximum(method_polynomial=analytical)

        return tangent_theta,max_theta

    def fit_linker(self,alpha):
        tangent = self.calculate_cartesian_boundary_points(alpha)
        # tangent = self.rotate_contour(tangent, self.v['x0'],self.v['y0'],self.v['theta0'])
        x_c = np.amax(tangent[:,0])
        y_c = np.amax(tangent[:,1])
        return x_c,y_c

    def ball_shape_fit(self, concavity_flag = False, x_c=0, y_c=0, tan_alpha=0, root_theta = None, a =50, b = 50, show_flag=False, clahe= False):
        """
        In this method, a fit is obtained after completing the shape of the ball (missing because of the ventouse).
        The fit parameters are then produced, which allow us to understand the physics of the problem.
        """
        STEP = np.pi/1800.
        if concavity_flag:
            try:
                self.find_concavity(self.thresholds['concavity_shrinker'], self.thresholds['concavity_offset'], a, b, show_flag=show_flag, clahe=clahe,vertical_negative_offset=self.thresholds['concavity_y_negative_offset'])
                concavity_thresh = 100
                zero = self.concavity_shape.shape[0]>concavity_thresh
                i = 1
                while not zero:
                    self.find_concavity(self.thresholds['concavity_shrinker'], self.thresholds['concavity_offset'], a - i, b - i, show_flag=show_flag,
                                        clahe=clahe,vertical_negative_offset = self.thresholds['concavity_y_negative_offset'])
                    if self.concavity_shape.shape[0]>concavity_thresh:
                        zero = True
                    elif (a-i)<=0 or (b-i) <=0:
                        raise IndexError
                    else:
                        i+=1


                self.draw_contour((0, 0, 0),1,contour=(self.concavity_shape[:,0],self.concavity_shape[:,1]))
                self.concavity_shape = self.rotate_contour(self.concavity_shape,self.v['x0'],self.v['y0'],self.v['theta0'])
                success = self.concavity_fitter(x_c, y_c,tan_alpha=tan_alpha)
                self.c = self.concavity_shape_parameters.valuesdict()
                theta_2 = root_theta
                self.fitted_concavity_shape = self.generate_fitted_concavity_shape(x_c,self.c)
                theta_external = np.arange(theta_2, np.pi/2., STEP)
                self.fitted_external_shape = self.generate_fitted_contour(self.v, self.thresholds['fit_parameters_nbr'], theta=theta_external)
                self.draw_contour((0, 0, 0), 1, contour=self.fitted_external_shape)
                self.draw_contour((0, 0, 0), 1, contour=self.fitted_concavity_shape)
                theta_external_2 = np.arange(-np.pi, np.pi, STEP)
                self.fitted_external_shape_2 = np.array(self.generate_fitted_contour(self.v,self.thresholds['fit_parameters_nbr'],theta=theta_external_2))
                # self.draw_contour((0,0,0),1,contour=self.fitted_external_shape_2)
                return success
            except IndexError:
                return False
        else:
            theta_external = np.arange(-np.pi, np.pi, STEP)
            self.fitted_external_shape = np.array(self.generate_fitted_contour(self.v,self.thresholds['fit_parameters_nbr'],theta=theta_external))
            self.draw_contour((0,0,0),1,contour=self.fitted_external_shape)
            return True

    def calculateR(self,parameters,data,nbr):
        """fit the shape depending on the number of parameters"""
        v = parameters.valuesdict()
        experimental_radius = np.sqrt((data[:,0]- v['x0'])**2+(data[:,1]-v['y0'])**2)
        v = parameters.valuesdict()
        theta = np.arctan2((data[:,1]-v['y0']),( data[:,0]-v['x0']))

        theoretical_radius = 0
        for i in xrange (0,nbr+1):
            name = 'a'+str(i)
            deltaR = v[name]
            theta_th = (np.sin(theta-v['theta0']))**i
            deltaR = deltaR * theta_th
            theoretical_radius  += deltaR
        theoretical_radius  = v['b0']* theoretical_radius
        return (experimental_radius-theoretical_radius)

    def circular(self,p):
        """circular fit"""
        v = p.valuesdict()
        experimental_R = np.sqrt((self.circle_ball_shape[:,0]- v['x0'])**2+ (self.circle_ball_shape[:,1]-v['y0'])**2)
        theta = np.arctan2((self.circle_ball_shape[:,1]-v['y0']),( self.circle_ball_shape[:,0]-v['x0']))
        theoretical_R = v['b0']*(1+v['a1']*np.sin(theta))
        return (experimental_R-theoretical_R)

    def crop_img(self,order = 1):
        """To be used in the beginning to reduce the size of the image to be treated"""
        if order == 1:
            x0 = self.thresholds["1crop_x0"]
            y0 = self.thresholds["1crop_y0"]
            h = self.thresholds["1crop_h"]
            w = self.thresholds["1crop_w"]
            self.in_img = self.in_img[y0:y0+h,x0:x0+w]
            self.out_img = np.copy(self.in_img)
        else:
            x0 = self.thresholds["2crop_x0"]
            y0 = self.thresholds["2crop_y0"]
            h = self.thresholds["2crop_h"]
            w = self.thresholds["2crop_w"]
            self.in_img = self.in_img[y0:y0+h,x0:x0+w]
            self.out_img = np.copy(self.in_img)

    def draw_contour(self,color,thickness,drawBallContour = False,fitFlag=False,contour=None,):
        """Draw the contour obtained by the ball_shape_fit"""
        if fitFlag:
            theta = self.roots_theta
            fitted_contour = 1
            for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
                name = 'a'+str(i)
                fitted_contour += self.v[name]*(np.sin(theta-self.v['theta0']))**i
            fitted_contour = self.v['b0']*fitted_contour
            x_derivative = self.v['x0']+fitted_contour*np.cos(theta)
            y_derivative = self.v['y0']+fitted_contour*np.sin(theta)


            for i in xrange(self.Xth.shape[0]):
                cv2.line(self.out_img,(self.Xth[i].astype(int),self.Yth[i].astype(int)),(self.Xth[i].astype(int),self.Yth[i].astype(int)),color,thickness)
            cv2.line(self.out_img,(self.v['x0'].astype(int),self.v['y0'].astype(int)),(self.v['x0'].astype(int),self.v['y0'].astype(int)),color, thickness)
            for i in xrange(self.roots_theta.shape[0]):
                cv2.line(self.out_img,(x_derivative[i].astype(int),y_derivative[i].astype(int)),(x_derivative[i].astype(int),y_derivative[i].astype(int)),color, thickness+2)
        if contour is None:
            if drawBallContour:
                Xexp = self.ball_shape[:,0]
                Yexp = self.ball_shape[:,1]
                for i in xrange(Xexp.shape[0]):
                    cv2.line(self.out_img,(Xexp[i].astype(int),Yexp[i].astype(int)),(Xexp[i].astype(int),Yexp[i].astype(int)),color,thickness)
        else:
            Xexp = contour[0]
            Yexp = contour[1]
            for i in xrange(Xexp.shape[0]):
                cv2.line(self.out_img,(Xexp[i].astype(int),Yexp[i].astype(int)),(Xexp[i].astype(int),Yexp[i].astype(int)),color,thickness)

    def draw_dot(self,color,thickness,dot =(0,0)):

        cv2.line(self.out_img,(dot[0].astype(int),dot[1].astype(int)),(dot[0].astype(int),dot[1].astype(int)),color, thickness)

    def max_height(self, spherique = False):
        if not spherique:
            max_height = 1
            for i in xrange(1,self.thresholds['fit_parameters_nbr']+1):
                if i%2==0:
                    max_height +=self.v["a"+str(i)]
            return max_height*self.v['b0']*self.thresholds['conversion']
        else:
            return self.v["b0"]*self.thresholds['conversion']

    def image_show(self):
        cv2.imshow('image',self.in_img)
        cv2.imshow('out',self.out_img)
        cv2.waitKey(1)

    def generate_fitted_contour(self, parameters, nbr, step=np.pi/1800.0, theta=None):
        if theta is None:
            theta = np.arange(-np.pi, np.pi, step)
        fitted_contour = 0
        derivate = 0
        for i in xrange (0,nbr+1):
            name = 'a'+str(i)
            # fitted_contour += parameters[name]*(np.sin(theta-parameters['theta0']))**i
            fitted_contour += parameters[name]*(np.sin(theta))**i
        Xth = fitted_contour*np.cos(theta)
        Yth = fitted_contour*np.sin(theta)
        fitted_centered_contour = np.ndarray((Yth.shape[0],2),dtype=Yth.dtype)
        fitted_centered_contour[:, 0] = Xth
        fitted_centered_contour[:, 1] = Yth
        fitted_contour = self.rotate_contour(fitted_centered_contour,0, 0, -self.v['theta0'])
        fitted_contour[:, 0] += self.v['x0']
        fitted_contour[:, 1] += self.v['y0']
        X = fitted_contour[:, 0]
        Y = fitted_contour[:, 1]

        return  X,Y
        # Xth = parameters['x0'] + fitted_contour*np.cos(theta)
        # Yth = parameters['y0'] + fitted_contour*np.sin(theta)
        # return Xth, Yth

    def max_width(self):

        max_width = (np.amax(self.Xth, axis=0)-self.v["x0"])*2
        theta_max_width = self.theta[np.argmax(self.Xth, axis=0)]
        return max_width*self.thresholds['conversion']/2.,theta_max_width

    def compute_external_volume(self, concavity_flag=False, phi0=0, R0=0, pix=False):
        "R0 is xc, phi0 angle of xc"
        if concavity_flag:
            if pix is not True:
                V_ext_sphere = 0.0
                b = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                for i in xrange(0, self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    b[i] = self.v[name]
                for i in xrange(self.thresholds["fit_parameters_nbr"]+1):
                    for j in xrange(self.thresholds["fit_parameters_nbr"]+1):
                        for k in xrange(self.thresholds["fit_parameters_nbr"]+1):
                            coefficients = b[i]*b[j]*b[k]
                            added = coefficients * ((1.-(np.cos(phi0))**(i+j+k+1))/(1.+i+j+k))
                            V_ext_sphere += added
                V_ext_sphere *= 2*(np.pi/3.)*self.thresholds['conversion']**3
                V_ext_calotte = 0.0
                # R0 = R0*
                for i in xrange(0, self.thresholds['degree_polynomial']+1,2):
                    name = 'a'+str(i)
                    c = self.c[name]
                    V_ext_calotte -= (1./(i+2.))*c*((R0)**(i+2))

                alpha = (np.pi/2.)-phi0
                tan = (1./3.)*np.tan(alpha)*R0**3
                V_ext_calotte += tan
                V_ext_calotte *= 2*np.pi*(self.thresholds['conversion']**3)
                return (V_ext_sphere+V_ext_calotte)
            else:
                V_ext_sphere = 0.0
                b = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                for i in xrange(0, self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    b[i] = self.v[name]
                for i in xrange(self.thresholds["fit_parameters_nbr"]+1):
                    for j in xrange(self.thresholds["fit_parameters_nbr"]+1):
                        for k in xrange(self.thresholds["fit_parameters_nbr"]+1):
                            V_ext_sphere += b[i]*b[j]*b[k]* ((1.-(np.cos(phi0))**(i+j+k))/(1.+i+j+k))

                V_ext_calotte = 0.0
                for i in xrange(0, self.thresholds['degree_polynomial']+1,2):
                    name = 'a'+str(i)
                    V_ext_calotte += (1./(2.+i))*self.v[name]*R0**(i+2)
                V_ext_calotte -= (1./3.)*np.tan(phi0)*R0**3
                V_ext_calotte *= -2*np.pi

                return (V_ext_sphere*2*np.pi/3. + V_ext_calotte)
        else:
            if pix is not True:
                V_ext_sphere = 0.0
                b = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                for i in xrange(0, self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    b[i] = self.v[name]
                for i in xrange(self.thresholds["fit_parameters_nbr"]+1):
                    for j in xrange(self.thresholds["fit_parameters_nbr"]+1):
                        for k in xrange(self.thresholds["fit_parameters_nbr"]+1):
                            V_ext_sphere += b[i]*b[j]*b[k]* ((1.+(-1.)**(i+j+k))/(1.+i+j+k))

                return (V_ext_sphere*2*np.pi/3.)*self.thresholds['conversion']**3
            else:
                V_ext_sphere = 0.0
                b = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                for i in xrange(0, self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    b[i] = self.v[name]
                for i in xrange(self.thresholds["fit_parameters_nbr"]+1):
                    for j in xrange(self.thresholds["fit_parameters_nbr"]+1):
                        for k in xrange(self.thresholds["fit_parameters_nbr"]+1):
                            V_ext_sphere += b[i]*b[j]*b[k]* ((1.+(np.cos(phi0))**(i+j+k))/(1.+i+j+k))

                return (V_ext_sphere*2*np.pi/3.)

    def gravity_center(self,concavity_flag=False, phi0=0, R0=0, pix=False):
        "DONE"
        if pix is not True:
            if concavity_flag:
                xg = self.v['x0']*self.thresholds['conversion']
                yg = self.v['y0']*self.thresholds['conversion']
                b = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                b[0] =  self.v['a0']
                for i in xrange(1, self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    b[i] = self.v[name]
                bp = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                bp[0] = self.v['a0'] - (self.thresholds['shell_thickness']/(self.v['b0']*self.thresholds['conversion']))
                for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    bp[i] = self.v[name]
                yg_ai = 0.0
                for i in xrange(self.thresholds["fit_parameters_nbr"]+1):
                    for j in xrange(self.thresholds["fit_parameters_nbr"]+1):
                        for k in xrange(self.thresholds["fit_parameters_nbr"]+1):
                            for l in xrange(self.thresholds["fit_parameters_nbr"]+1):
                                a = b[i]
                                L = b[l]
                                b1 = b[i]*b[j]*b[k]*b[l]
                                b2 = bp[i]*bp[j]*bp[k]*bp[l]
                                b_i = (b[i]*b[j]*b[k]*b[l]-bp[i]*bp[j]*bp[k]*bp[l])
                                cos = (1.-(np.cos(phi0))**(i+j+k+l+2.))/(2.+i+j+k+l)
                                yg_ai += b_i*cos
                yg_ai *= (np.pi/2.)*self.thresholds['conversion']**4
                d = self.thresholds['shell_thickness']/self.thresholds['conversion']
                v = []
                for i in xrange(0,self.thresholds['degree_polynomial']+1,2):
                    name = 'a'+str(i)
                    v.append(self.c[name])
                vp = []
                vp.append(self.c['a0']+d)
                for i in xrange(2,self.thresholds['degree_polynomial']+1,2):
                    name = 'a'+str(i)
                    vp.append(self.c[name])
                phi0 = (np.pi/2.)-phi0
                R1 = R0-(d*np.cos(phi0))
                y1 = (-1/2.)*(R0**2)*v[0]**2+(-1/2.)*(R0**4)*v[0]*v[1]+(-1/6.)*(R0**6)*v[1]**2+(-1/3.)*(R0**6)*v[0]*v[2]+(-1/4.)*(R0**8)*v[1]*v[2]+(-1/10.)*(R0**10)*v[2]**2+(-1/4.)*(R0**8)*v[0]*v[3]+(-1/5.)*(R0**10)*v[1]*v[3]+(-1/6.)*(R0**12)*v[2]*v[3]+(-1/14.)*(R0**14)*v[3]**2+(1/4.)*(R0**4)*(np.tan(phi0))**2
                y2 = (-1/2.)*(R1**2)*vp[0]**2+(-1/2.)*(R1**4)*vp[0]*v[1]+(-1/6.)*(R1**6)*v[1]**2+(-1/3.)*(R1**6)*vp[0]*v[2]+(-1/4.)*(R1**8)*v[1]*v[2]+(-1/10.)*(R1**10)*v[2]**2+(-1/4.)*(R1**8)*vp[0]*v[3]+(-1/5.)*(R1**10)*v[1]*v[3]+(-1/6.)*(R1**12)*v[2]*v[3]+(-1/14.)*(R1**14)*v[3]**2+(1/4.)*(R1**4)*(np.tan(phi0))**2
                yg_ak = np.pi*(y1-y2)*self.thresholds['conversion']**4
                yg_ai += yg_ak
                yg_ai *= 1./self.thresholds['shell_volume']
                xg += yg_ai*np.sin(self.v['theta0'])
                yg += yg_ai*np.cos(self.v['theta0'])
                return xg, yg

            else:
                xg = self.v['x0']*self.thresholds['conversion']
                yg = self.v['y0']*self.thresholds['conversion']
                b = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                b[0] =  self.v['a0']
                for i in xrange(1, self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    b[i] = self.v[name]
                bp = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                bp[0] = self.v['a0'] - (self.thresholds['shell_thickness']/(self.v['b0']*self.thresholds['conversion']))
                for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    bp[i] = self.v[name]
                yg_ai = 0.0
                for i in xrange(self.thresholds["fit_parameters_nbr"]+1):
                    for j in xrange(self.thresholds["fit_parameters_nbr"]+1):
                        for k in xrange(self.thresholds["fit_parameters_nbr"]+1):
                            for l in xrange(self.thresholds["fit_parameters_nbr"]+1):
                                a = b[i]
                                L = b[l]
                                yg_ai += (b[i]*b[j]*b[k]*b[l]-bp[i]*bp[j]*bp[k]*bp[l])*(1.-(-1)**(i+j+k+l+2.))/(2.+i+j+k+l)
                yg_ai *= (np.pi/2.)*self.thresholds['conversion']**4
                yg_ai *= 1./self.thresholds['shell_volume']
                xg = xg + yg_ai*np.sin(self.v['theta0'])
                yg = yg + yg_ai*np.cos(self.v['theta0'])
                return xg, yg
        else:
            if concavity_flag:
                xg = self.v['x0']
                yg = self.v['y0']
                b = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                b[0] =  self.v['a0']
                for i in xrange(1, self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    b[i] = self.v[name]
                bp = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                bp[0] = self.v['a0'] - (self.thresholds['shell_thickness']/(self.v['b0']*self.thresholds['conversion']))
                for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    bp[i] = self.v[name]
                yg_ai = 0.0
                for i in xrange(self.thresholds["fit_parameters_nbr"]+1):
                    for j in xrange(self.thresholds["fit_parameters_nbr"]+1):
                        for k in xrange(self.thresholds["fit_parameters_nbr"]+1):
                            for l in xrange(self.thresholds["fit_parameters_nbr"]+1):
                                a = b[i]
                                L = b[l]
                                yg_ai += (b[i]*b[j]*b[k]*b[l]-bp[i]*bp[j]*bp[k]*bp[l])*(1.-(np.cos(phi0))**(i+j+k+l+2.))/(2.+i+j+k+l)
                yg_ai *= (np.pi/2.)
                d = self.thresholds['shell_thickness']/self.thresholds['conversion']
                v = []
                for i in xrange(0,self.thresholds['degree_polynomial']+1,2):
                    name = 'a'+str(i)
                    v.append(self.c[name])
                vp = []
                vp.append(self.c['a0']+d)
                for i in xrange(2,self.thresholds['degree_polynomial']+1,2):
                    name = 'a'+str(i)
                    vp.append(self.c[name])
                phi0 = (np.pi/2.)-phi0
                R1 = R0-(d*np.cos(phi0))
                y1 = (-1/2.)*(R0**2)*v[0]**2+(-1/2.)*(R0**4)*v[0]*v[1]+(-1/6.)*(R0**6)*v[1]**2+(-1/3.)*(R0**6)*v[0]*v[2]+(-1/4.)*(R0**8)*v[1]*v[2]+(-1/10.)*(R0**10)*v[2]**2+(-1/4.)*(R0**8)*v[0]*v[3]+(-1/5.)*(R0**10)*v[1]*v[3]+(-1/6.)*(R0**12)*v[2]*v[3]+(-1/14.)*(R0**14)*v[3]**2+(1/4.)*(R0**4)*(np.tan(phi0))**2
                y2 = (-1/2.)*(R1**2)*vp[0]**2+(-1/2.)*(R1**4)*vp[0]*v[1]+(-1/6.)*(R1**6)*v[1]**2+(-1/3.)*(R1**6)*vp[0]*v[2]+(-1/4.)*(R1**8)*v[1]*v[2]+(-1/10.)*(R1**10)*v[2]**2+(-1/4.)*(R1**8)*vp[0]*v[3]+(-1/5.)*(R1**10)*v[1]*v[3]+(-1/6.)*(R1**12)*v[2]*v[3]+(-1/14.)*(R1**14)*v[3]**2+(1/4.)*(R1**4)*(np.tan(phi0))**2
                yg_ak = np.pi*(y1-y2)
                yg_ai += yg_ak
                yg_ai *= (self.thresholds['conversion']**3)/self.thresholds['shell_volume']
                xg = xg + yg_ai*np.sin(self.v['theta0'])
                yg = yg + yg_ai*np.cos(self.v['theta0'])
                return xg, yg
            else:
                xg = self.v['x0']
                yg = self.v['y0']
                b = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                b[0] =  self.v['a0']
                for i in xrange(1, self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    b[i] = self.v[name]
                bp = np.zeros(self.thresholds['fit_parameters_nbr']+1)
                bp[0] = self.v['a0'] - (self.thresholds['shell_thickness']/(self.v['b0']*self.thresholds['conversion']))
                for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
                    name = 'a'+str(i)
                    bp[i] = self.v[name]
                yg_ai = 0.0
                for i in xrange(self.thresholds["fit_parameters_nbr"]+1):
                    for j in xrange(self.thresholds["fit_parameters_nbr"]+1):
                        for k in xrange(self.thresholds["fit_parameters_nbr"]+1):
                            for l in xrange(self.thresholds["fit_parameters_nbr"]+1):
                                a = b[i]
                                L = b[l]
                                yg_ai += (b[i]*b[j]*b[k]*b[l]-bp[i]*bp[j]*bp[k]*bp[l])*(1.-(-1)**(i+j+k+l+2.))/(2.+i+j+k+l)
                yg_ai *= (np.pi/2.)*(self.thresholds['conversion']**3)/self.thresholds['shell_volume']
                xg = xg + yg_ai*np.sin(self.v['theta0'])
                yg = yg + yg_ai*np.cos(self.v['theta0'])
                return xg, yg

    def read_parameters(self,keys,line):
        """a tester"""
        values = line.split("\t")
        self.v = {}
        for i in xrange(2,len(values)):
            self.v[keys[i]] = float(values[i])

    def write_raw_data(self,output_name):
        f_raw_data = open(output_name,"w")
        f_raw_data.writelines("X"+"\t"+"Y")
        for i in xrange(self.ball_shape.shape[0]):
                f_raw_data.writelines("\n")
                f_raw_data.writelines("{0:.5e}".format(self.ball_shape[i][0])+'\t')
                f_raw_data.writelines("{0:.5e}".format(self.ball_shape[i][1])+'\t')
        f_raw_data.close()

    def plot_fit(self):
        theta = np.arange(-np.pi, np.pi, np.pi/1800.0)
        fitted_contour = 1
        derivative_contour = 0
        for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
            name = 'a'+str(i)
            fitted_contour += self.v[name]*(np.sin(theta-self.v['theta0']))**i
            derivative_contour += (i*self.v[name]*(np.sin(theta-self.v['theta0']))**(i-1))
        # derivative_contour = derivative_contour*np.cos(theta-self.v['theta0'])
        zero = np.zeros(derivative_contour.shape)
        plt.plot(theta,fitted_contour)
        plt.plot(theta,derivative_contour)
        plt.plot(theta,zero)
        plt.show()

    def tangent(self,method_polynomial = True):
        "repere absolu"
        theta_correction = 0
        if method_polynomial:
            fitted_coefficients = []
            for i in xrange (0,self.thresholds['fit_parameters_nbr']+1):
                name = 'a'+str(i)
                fitted_coefficients = np.append(fitted_coefficients,(i+1)*self.v[name])
            fitted_coefficients = fitted_coefficients[::-1]
            x= np.arange(-1,0,0.0001)
            y = np.polyval(fitted_coefficients,x)
            plt.legend()
            plt.show()
            roots = np.roots(fitted_coefficients)
            roots = roots.real[abs(roots.imag)<1e-5]
            roots = np.amax(np.arcsin(roots[np.where(abs(roots)<=1)]))
        else:
            step = np.pi/1800.
            theta = np.arange(-np.pi/2.,0, step)
            tangent1 = self.v['a0']
            tangent2 = 0
            for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
                name = 'a'+str(i)
                tangent1 += self.v[name]*np.sin(theta)**i
                tangent2 += i*self.v[name]*np.sin(theta)**i
            tangent1= abs(tangent1)
            tangent2= abs(tangent2)
            tangent_array = tangent1-tangent2
            roots = np.amin(abs(tangent_array))
            index = np.argmin(abs(tangent_array),axis=0)
            roots = theta[index]

        theta0 = self.v['theta0']
        roots_theta = roots+theta_correction
        theta_conjugate = -(np.pi-abs(roots_theta))
        roots_theta = np.append(roots_theta,theta_conjugate)
        roots_theta += 0
        return roots_theta

    def R_maximum(self,method_polynomial = True):
        theta_correction = 0
        if method_polynomial:
            fitted_coefficients = []
            for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
                name = 'a'+str(i)
                fitted_coefficients = np.append(fitted_coefficients,(i)*self.v[name])
            fitted_coefficients = fitted_coefficients[::-1]
            x= np.arange(-1,0,0.0001)
            y = np.polyval(fitted_coefficients,x)
            roots = np.roots(fitted_coefficients)
            roots = roots.real[abs(roots.imag)<1e-5]
            roots = np.amax(np.arcsin(roots[np.where(abs(roots)<=1)]))

        else:
            theta = np.arange(-np.pi/2,0, np.pi/1800.)
            maximum = 0
            for i in xrange (0,self.thresholds['fit_parameters_nbr']+1):
                name = 'a'+str(i)
                maximum += self.v[name]*np.sin(theta)**(i)
            roots = np.amax(abs(maximum))
            # index = np.where(abs(maximum)<1e-3)
            index = np.argmax(abs(maximum),axis=0)
            roots = theta[index]

        theta0 = self.v['theta0']
        roots_theta = roots+theta_correction
        theta_conjugate = -(np.pi-abs(roots_theta))
        roots_theta = np.append(roots_theta,theta_conjugate)
        roots_theta += 0
        return roots_theta

    def tangent_minimal(self,theta_min,theta_max,epsilon):
        'epsilon negative'
        step = np.pi/1800.
        if theta_min[0]<theta_max[0]:
            theta = np.arange(theta_min[0],theta_max[0],step)
        else:
            theta_min = -5*np.pi/12.
            theta_max = -np.pi/4.
            theta = np.arange(theta_min,theta_max,step)

        tangent1 = self.v['a0']
        tangent2 = 0
        for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
            name = 'a'+str(i)
            tangent1 += self.v[name]*np.sin(theta)**i
            tangent2 += i*self.v[name]*np.sin(theta)**(i-1)
        tangent2 *= np.cos(theta)
        tangent3 = np.tan(theta+epsilon)
        y1 = (tangent1/tangent2)
        y2 = tangent3
        y3 = np.tan(theta)
        y = abs(y1+y2)
        # plt.plot(theta,y1,'g^',label='tangente')
        # plt.plot(theta,y2,'r--',label='theta+epsilon')
        # plt.plot(theta,y3,'b-',label='theta')
        # plt.plot(theta,y,'mo',label='pratic')
        # plt.legend()
        # plt.show()
        index = np.argmin(abs(y),axis=0)
        theta_min1 = theta[index]
        print "minimum value tangent: " +str(np.amin(abs(y),axis=0))
        print "Theta minimum value tangent: "+str(theta_min1)
        return theta_min1

    def calculate_cartesian_boundary_points(self,theta):
        ""
        fitted_contour = 0
        for i in xrange (0,self.thresholds['fit_parameters_nbr']+1):
            name = 'a'+str(i)
            fitted_contour += self.v[name]*(np.sin(theta))**i
        fitted_contour = self.v['b0']*fitted_contour
        x_derivative = fitted_contour*np.cos(theta)
        y_derivative = fitted_contour*np.sin(theta)
        tangent = np.ndarray((2, 2), dtype=x_derivative.dtype)
        tangent[:, 0] = np.array(x_derivative)
        tangent[:, 1] = np.array(y_derivative)
        return tangent

    def calculate_tangent_alpha(self,epsilon):

        # tangent1 = self.v['a0']
        # tangent2 = 0
        # for i in xrange (1,self.thresholds['fit_parameters_nbr']+1):
        #     name = 'a'+str(i)
        #     tangent1 += self.v[name]*np.sin(theta)**i
        #     tangent2 += i*self.v[name]*np.sin(theta)**(i-1)
        # tangent2 *= np.cos(theta)
        # alpha = np.arctan(tangent1/tangent2)
        # beta = np.pi-(alpha+theta)
        # print "beta is:"+ str(beta)
        return -np.tan(epsilon)

    def calculate_r_theta(self):
        ""
        theta = self.roots_theta
        fitted_contour = 0
        for i in xrange (0,self.thresholds['fit_parameters_nbr']+1):
            name = 'a'+str(i)
            fitted_contour += self.v[name]*(np.sin(theta-self.v['theta0']))**i
        fitted_contour = self.v['b0']*fitted_contour
        r_c = fitted_contour
        theta = self.roots_theta - self.v['theta0']
        return r_c,theta

    def rotate_contour(self,contour,x0,y0,theta0):
        # plt.plot(contour[:,0],contour[:,1],'.')
        # plt.show()
        rotated_contour = np.zeros(contour.shape,contour.dtype)
        contour[:,0] -= int(x0)
        contour[:,1] -= int(y0)
        rotated_contour[:, 0] = contour[:,0]*np.cos(theta0)+contour[:,1]*np.sin(theta0)
        rotated_contour[:, 1] = -contour[:,0]*np.sin(theta0)+contour[:,1]*np.cos(theta0)
        # plt.plot(rotated_contour[:,0],rotated_contour[:,1],'*')
        # plt.show()
        return rotated_contour

    def calculate_concavity(self,parameters):
        v = parameters.valuesdict()
        x = self.concavity_shape[:,0]
        model =0
        for i in xrange (0,7,2):
            name = 'a'+str(i)
            model  += v[name]*x**i
        return (model-self.concavity_shape[:,1])

    def generate_fitted_concavity_shape(self, a, c):
        x_sym = np.arange(0, a, 0.001)
        x = np.arange(0, a, 0.001)
        # coeff = [c['a4'], 0, c['a2'], 0, c['a0']]
        coeff = [c['a6'], 0, c['a4'], 0, c['a2'], 0, c['a0']]
        y = np.polyval(coeff, x)


        # ymax = np.amax(y,axis=0)
        # ymin = np.amin(y,axis=0)
        # print "YMAAAAAAAAAX"+str(ymax)
        # x = np.concatenate((x,np.arange(ymin,abs(ymax),1)*0),axis=0)
        # y = np.concatenate((y,np.arange(ymin,abs(ymax),1)),axis=0)
        # plt.plot(x,y)
        # plt.show()
        fitted_centered_contour = np.ndarray((y.shape[0],2),dtype=y.dtype)
        fitted_centered_contour[:, 0] = x
        fitted_centered_contour[:, 1] = y
        fitted_contour = self.rotate_contour(fitted_centered_contour,0, 0, -self.v['theta0'])
        fitted_contour[:, 0] += self.v['x0']
        fitted_contour[:, 1] += self.v['y0']
        X = fitted_contour[:, 0]
        Y = fitted_contour[:, 1]

        return X,Y

    def save(self, path_out=None):
        """Only make sens if draw contour is used"""
        if self.out_name is None:
            cv2.imwrite(self.out_name,self.out_img)

        else:
            if path_out is not None:
                cv2.imwrite(path_out,self.out_img)
            else:
                print("IT IS NOT POSSIBLE TO SAVE THE IMAGE.")

    def convert_parameters(self):
        """Has to be appiled before saving parameters, after computing volumes."""
        self.v['b0'] *=self.thresholds['conversion']
        self.v['x0'] *=self.thresholds['conversion']
        self.v['y0'] *=self.thresholds['conversion']

    def invert_convert_parameters(self):
        """Has to be appiled before saving parameters, after computing volumes."""
        self.v['b0'] /=self.thresholds['conversion']
        self.v['x0'] /=self.thresholds['conversion']
        self.v['y0'] /=self.thresholds['conversion']

    def transform_to_polar(self):
        theta = np.arange(-np.pi,np.pi,np.pi/1800.)
        a0 = self.c['a0']
        a2 = self.c['a2']
        a4 = self.c['a4']
        R11210 = 2*a2**3+27*a4-72*a0*a2*a4
        R11211 = (2*a2**3-27*a4-72*a0*a2*a4)*np.cos(2*theta)
        R1121 = (R11210+R11211)**2
        R1120 = -16*(((a2**2)+12*a0*a4)**3)*(np.cos(theta))**4
        R112 = np.sqrt((np.cos(theta)**8)*(R1120+R1121))
        R110 = 4*(a2**3)*np.cos(theta)**6
        R111 = -144*a0*a2*a4*np.cos(theta)**6
        R113 = 54*a4*(np.cos(theta)**4)*(np.sin(theta)**2)
        R10 = -4*a2*(1./np.cos(theta))**2
        R11 = (2**(1./3.))*(1./np.cos[theta])**4*(R110+R111+R112+R113)**(1./3.)


        R1210 = 2*a2**3
        R1211 = -108*a0*a2*a4
        R1212 = 6*(a2**3)*np.cos(2*theta)
        R1213 = -252*a0*a2*a4*np.cos(2*theta)
        R1214 = 6*(a2**3)*np.cos(2*theta)**2
        R1215 = -144*a0*a2*a4*np.cos(2*theta)**2
        R1216 = 2*(a2**3)*np.cos(2*theta)**3
        R1217 = 4*R112
        R1218 = -36*a0*a2*a4*np.cos(4*theta)
        R1219 = 81*a4*np.sin(theta)**2
        R121A = 108*a4*np.cos(2*theta)*np.sin(theta)**2
        R121B = 27*a4*np.cos(4*theta)*np.sin(theta)**2
        R121C = -9*a0*a2*a4*(1./np.sin(2*theta))*np.sin(8*theta)

        R120 = 4*(2**(1./3.))*(a2**2+12*a0*a4)
        R121 = (R1210+R1211+R1212+R1213+R1214+R1215+R1216+R1217+R1218+R1219+R121A+R121B+R121C)**(1./3.)
        R12 = R120/R121

        R1 = -np.sqrt((1./a4)*(R10+R11+R12))

        R20 = (2./a4)*R10
        R21 = -(1./a4)*R11
        R220 = -R120
        R221 = a4*R121
        R22 = R220/R221
        R230 = -(12*np.sqrt(6)*(1./np.cos(theta)**3)*np.tan(theta))
        R231 = np.sqrt(a4*(R10+R11+R12))
        R23 = R230/R231
        R2 = np.sqrt(R20+R21+R22+R23)

        R = 1./(2*np.sqrt(6))* (R1+R2)

    def get_down_point(self):
        """relatively to the croping parameters, values in pixel"""
        y_bas = np.amax(self.ball_shape[:,1],axis=0)
        return y_bas*self.thresholds['conversion']

    def rotate_image(self):
        (h, w) = self.in_img.shape[:2]
        center = (w / 2, h / 2)

        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated = cv2.warpAffine(self.in_img, M, (w, h))
        self.in_img = np.copy(rotated)
        self.out_img = np.copy(rotated)
