import numpy as np
import cv2
import scipy.interpolate as inter
from scipy.signal import savgol_filter
import lmfit
from matplotlib import pyplot as plt

class ShapePIV():
    '''
    Path_in is the original image path.
    Path_out is the output path of the processed image.
    whiteOnBlack means that -in the original THRESHOLDED image- background is black and lines are white, usually it's the opposite.
    img is a constructor by copy "surcharge"
    Thresholds are a dictionnary of the different thresholds to be used in the image process.

    '''
    def __init__(self, path_in,  path_out=None, thresholds=None,img=None,concavity=False):
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
        self.in_img_process = self.crop_img(self.in_img)
        self.pt_up_left = self.thresholds['crop_y0'],self.thresholds['crop_x0']
        # cv2.imshow("init",self.in_img_process)
        # cv2.waitKey()
        if concavity:
            h,w = self.in_img_process.shape
            self.h_a = int(self.thresholds['height_ratio']* h)
            self.in_img_a= self.in_img_process[:self.h_a,:]
            # cv2.imshow("a",self.in_img_a)
            # cv2.waitKey()
            self.in_img_b= self.in_img_process[self.h_a:,:]
            # cv2.imshow("b",self.in_img_b)
            # cv2.waitKey()
        self.concavity_flag = concavity

    def load(self):
        self.in_img = cv2.imread(self.in_name,0)

    def median_blur(self,img=None,sigma=3):
        img = cv2.medianBlur(img,sigma)
        return img

    def gaussian_blur(self,img=None,sigma=3):
        img = cv2.GaussianBlur(img,(sigma,sigma),0)
        return img

    def guassian_mean_threshold(self,img=None,threshold=3,sigma = 2):
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,threshold,sigma)
        return thresh

    def neighboring_array(self,point,main_array,blocksize):


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

    def get_xMax_values(self,array,filterblocksize,density_threshold,x_offset=0):
        list_y = []
        for i in xrange(np.amin(array[0:,1]),np.amax(array[0:,1])):
            yi = np.where(array[:,1]==i)

            x_positions = array[yi]

            try:
                yi_xmax = np.argmax(x_positions[0:,0])
                while True:
                    yi_xmax = np.argmax(x_positions[:,0])
                    density = self.neighboring_array(x_positions[yi_xmax],array,filterblocksize)
                    if density > density_threshold:
                        break
                    else:
                        x_positions = x_positions[:-1,:]

                x_positions[yi_xmax][0]+=x_offset

                list_y.append(x_positions[yi_xmax])
            except ValueError:
                pass
        return np.array(list_y)

    def get_nonzero(self,edges):
        'in image format index y:x'
        index = np.array(np.nonzero(edges),dtype=np.int32)
        at = index[0].copy()
        index[0] = index[1]
        index[1] = at
        index = index.transpose()
        return index

    def to_image_format(self,array):
        if array.shape[0]< array.shape[1]:
            at = array[0].copy()
            array[0] = array[1]
            array[1] = at
        else:
            x_axis = array[:,0].copy()
            array[:,0] = array[:,1]
            y_axis = array[:,0]
            array[:,1] = x_axis
        return array

    def scan_boundaries(self,img,density_threshold,blocksize_x,blocksize_y,step=1, horizontal_flag=True,neutral_fiber=-1):
        if horizontal_flag:
            width  = img.shape[1]
            if neutral_fiber<0:
                y_initial = int(blocksize_y/2.)
            else:
                y_initial = neutral_fiber
            accumulated_x = 0
            density_list = []
            axis_list = []
            while accumulated_x<width:
                temp_sub_image = img[y_initial-int(blocksize_y/2.):y_initial+int(blocksize_y/2.),accumulated_x:accumulated_x+blocksize_x]
                density_list.append(np.count_nonzero(temp_sub_image))
                axis_list.append(accumulated_x+int(blocksize_x/2.))
                accumulated_x+=step
            temp_sub_image = img[y_initial-int(blocksize_y/2.):y_initial+int(blocksize_y/2.),-blocksize_x:-1]
            density_list.append(np.count_nonzero(temp_sub_image))
            axis_list.append(width-int(blocksize_x/2.))
            # return density_list,axis_list,y_initial
            density_array = np.array(density_list,dtype=np.float)/float(blocksize_x*blocksize_y)
            try:
                if density_threshold>1 or density_threshold<0:
                    raise ValueError

                density_index = np.argwhere(density_array<density_threshold)[0]
                x_centered = axis_list[density_index]
                return y_initial,x_centered
            except ValueError:
                print 'density must be in [0,1], it\' a probability, fucker!'


        else:
            height = img.shape[0]
            width  = img.shape[1]
            if neutral_fiber<0:
                x_initial = width-int(blocksize_y/2.)
            else:
                x_initial = neutral_fiber
            accumulated_y = 0
            density_list = []
            axis_list = []
            while accumulated_y<height:
                temp_sub_image = img[accumulated_y:accumulated_y+blocksize_y,x_initial-int(blocksize_x/2.):x_initial+int(blocksize_x/2.)]
                density_list.append(np.count_nonzero(temp_sub_image))
                axis_list.append(accumulated_y+int(blocksize_x/2.))
                accumulated_y+=step
            temp_sub_image = img[-blocksize_y:-1,x_initial-int(blocksize_x/2.):x_initial+int(blocksize_x/2.)]
            density_list.append(np.count_nonzero(temp_sub_image))
            axis_list.append(width-int(blocksize_x/2.))
            density_array = np.array(density_list,dtype=np.uint8)/float(blocksize_x*blocksize_y)
            try:
                if density_threshold>1 or density_threshold<0:
                    raise ValueError
                density_index = np.argwhere(density_array>density_threshold)[0]

                y_centered = axis_list[density_index]
                return y_centered,x_initial
            except IndexError:
                print 'density must be in [0,1], it\' a probability, fucker!'
                print np.amax(density_array)

    def define_crop(self,img_wb):
        try:

            if np.where(img_wb[np.nonzero(img_wb)]<255)[0].size>0:
                raise ValueError
            h_y, h_x = self.scan_boundaries(img_wb,self.thresholds['density_thresh_h'],self.thresholds['blocksize_x_h'],self.thresholds['blocksize_y_h'],step=self.thresholds['step_h'])
            v_y, v_x = self.scan_boundaries(img_wb,self.thresholds['density_thresh_v'],self.thresholds['blocksize_x_v'],self.thresholds['blocksize_y_v'],step=self.thresholds['step_v'],horizontal_flag= False)
            h_y -= int(self.thresholds['blocksize_y_h']/2.)
            h_x -= int(self.thresholds['offset_h'])
            v_y -= int(self.thresholds['offset_v'])
            v_x -= int(self.thresholds['blocksize_x_v']/2.)

            return self.in_img[h_y:v_y,h_x:v_x],(h_y,h_x),(v_y,v_x)

        except ValueError:
            print 'Image is not binary,FUCKER!'
            return None

    def shape_analyse(self,x0,cartesian=False):
        if self.concavity_flag:
            edges_a = self.image_preparation(self.in_img_a,self.thresholds['gaussian_blur_a'],canny_flag=True,thresh_adaptive_level=self.thresholds['thresh_adaptive_level_a'],sigma_thresh=self.thresholds['thresh_adaptive_sigma_a'],canny_a=self.thresholds['canny_a_a'],canny_b=self.thresholds['canny_b_a'])
            self.show("A",edges_a)
            index_a = self.get_nonzero(edges_a)
            y_array_a = self.get_xMax_values(index_a,self.thresholds['filter_blocksize_a'],self.thresholds['density_threshold'],self.thresholds['x_offset_a'])
            y_array_a = self.to_image_format(y_array_a)
            y_array_a[:,1] = savgol_filter(y_array_a[:,1],self.thresholds['savgol_window_size_a'],self.thresholds['pol_order_a'])

            edges_b = self.image_preparation(self.in_img_b,self.thresholds['gaussian_blur_b'],canny_flag=True,thresh_adaptive_level=self.thresholds['thresh_adaptive_level_b'],sigma_thresh=self.thresholds['thresh_adaptive_sigma_b'],canny_a=self.thresholds['canny_a_b'],canny_b=self.thresholds['canny_b_b'])
            self.show("B",edges_b)

            index_b = self.get_nonzero(edges_b)
            y_array_b = self.get_xMax_values(index_b,self.thresholds['filter_blocksize_b'],self.thresholds['density_threshold'],self.thresholds['x_offset_b'])
            y_array_b = self.to_image_format(y_array_b)
            y_array_b[:,1] = savgol_filter(y_array_b[:,1],self.thresholds['savgol_window_size_b'],self.thresholds['pol_order_b'])
            y_array_b[:,0] += self.h_a
            array_y = np.concatenate((y_array_a,y_array_b),axis=0)
            # print y_array_b[0]
            # y_high = y_array_b[np.argmin(y_array_b[:,0])]
            # y= np.arange(0,y_high[0]+0.1,0.1)
            # a= (y_high[1]-(y_high[1]+self.thresholds['array_a_x_offset']))
            # alpha = (y_high[1]-(y_high[1]+self.thresholds['array_a_x_offset']))/float((y_high[0]))
            # beta = y_high[1]+self.thresholds['array_a_x_offset']
            # x = alpha*y+beta
            # y_array_a = np.ndarray((x.shape[0],2))
            # y_array_a[:,0]=y
            # y_array_a[:,1]=x
            # ind = np.lexsort((y_array_a[:,1],y_array_a[:,0]))
            # y_array_a = y_array_a[ind]
            # array_y = np.concatenate((y_array_a,y_array_b),axis=0)


        else:
            edges = self.image_preparation(self.in_img_process,self.thresholds['gaussian_blur_whole'],canny_flag=False,thresh_adaptive_level=self.thresholds['thresh_adaptive_level_whole'],sigma_thresh=self.thresholds['thresh_adaptive_sigma_whole'],canny_a=self.thresholds['canny_a_whole'],canny_b=self.thresholds['canny_b_whole'])
            # cv2.imshow("whole",edges)
            # cv2.waitKey(1000)
            index = self.get_nonzero(edges)
            array_y = self.get_xMax_values(index,self.thresholds['filter_blocksize'],self.thresholds['density_threshold'],self.thresholds['x_offset_whole'])
            array_y = self.to_image_format(array_y)
            array_y[:,1] = savgol_filter(array_y[:,1],self.thresholds['savgol_window_size_whole'],self.thresholds['pol_order_whole'])




        # self.complete_array = array_y
        # self.complete_array = self.to_image_format(self.complete_array)
        # self.external_shape_fit()
        # self.color_path(255,edges,array_y)
        self.complete_array = self.get_contour_to_mask(array_y)
        self.complete_array = self.to_image_format(self.complete_array)
        if self.concavity_flag:
            if cartesian:
                self.complete_array = self.to_image_format(array_y)
                sucess =self.external_shape_fit(cartesian=True)
                self.v ={}
                self.v["x0"],self.v["y0"] = 0,0
            else:
                sucess =self.external_shape_fit(fit_nbr=self.thresholds['fit_parameters_nbr_separated'],x0=x0)
        else:
            sucess =self.external_shape_fit(fit_nbr=self.thresholds['fit_parameters_nbr_whole'],x0=x0)
        # array_y[:,1] +=self.pt_up_left[1]
        # self.fitted_external_shape[:,0] -= self.pt_up_left[1]
        if sucess:
            self.fitted_external_shape_final = self.get_contour_to_mask(self.fitted_external_shape)
        else:
            self.fitted_external_shape_final = self.to_image_format(self.fitted_external_shape_final)
            self.fitted_external_shape_final = self.get_contour_to_mask(self.complete_array)

        self.fitted_external_shape_final[:,1] += self.pt_up_left[1]
        # self.fitted_external_shape_final = self.to_image_format(self.fitted_external_shape_final)
        self.mask_img(self.in_img,self.fitted_external_shape_final)
        return self.v["x0"],self.v["y0"]

    def image_preparation(self,img, gaussianblur = 3,mediablur_flag =False, medianblur = 3,canny_flag = True,thresh_adaptive_level = 3, sigma_thresh = 2,canny_a =1,canny_b =1):
        img = self.gaussian_blur(img=img,sigma=gaussianblur)
        if mediablur_flag:
            img = self.median_blur(img,medianblur)
        if canny_flag:
            edges = cv2.Canny(img,canny_a,canny_b)
        else:
            edges = self.guassian_mean_threshold(img,thresh_adaptive_level,sigma_thresh)
        return edges

    def color_path(self,color,img,array,RGB =False):
        if RGB:
            try:
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            except cv2.error:
                pass
        pti_1=array[0][1].astype(int),0
        pti=array[0][1].astype(int),array[0][0].astype(int)
        cv2.line(img,pti_1,pti,color,1)
        for i in xrange(1,array.shape[0]):
            pti_1=array[i-1][1].astype(int),array[i-1][0].astype(int)
            pti=array[i][1].astype(int),array[i][0].astype(int)
            cv2.line(img,pti_1,pti,color,1)

    def get_contour_to_mask(self,array):
        output_img = np.zeros(self.in_img_process.shape, dtype=self.in_img.dtype)
        self.color_path(255,output_img,array)
        self.show("final",output_img)
        output_array = self.get_nonzero(output_img)
        output_array = self.to_image_format(output_array)
        return output_array

    def mask_img(self,img,array):

            for i in xrange(array.shape[0]):
                try:
                    x_temp = array[i][1].astype(int)
                    y_temp = array[i][0].astype(int)
                    # print y_temp,x_temp
                    img[y_temp,x_temp:]=0
                except IndexError:
                    pass


    def save(self,output_file,img):
        cv2.imwrite(output_file,img)

    def show(self,winname,img):
        cv2.imshow(winname,img)
        cv2.waitKey(100)

    def external_shape_fitter(self,x0,theta_free=False,verbose=False,fit_nbr=1):
        max_external_parameters_value = self.thresholds['max_external_parameters_value']
        min_external_parameters_value = self.thresholds['min_external_parameters_value']
        self.external_shape_parameters = lmfit.Parameters()
        self.external_shape_parameters.add('b0', value =1,vary = False )
        self.external_shape_parameters.add('x0',value = x0[0],max = x0[0]+max_external_parameters_value)
        self.external_shape_parameters.add('y0',value = x0[1],max = x0[1]+max_external_parameters_value)
        if theta_free is not True:
            self.external_shape_parameters.add('theta0', value=0.0,min=(0.001)*np.pi, max=(0.03)*np.pi)
        else:
            self.external_shape_parameters.add('theta0', value=0.0,vary=False)
        for i in xrange(0,fit_nbr+1):
            name = 'a'+str(i)
            self.external_shape_parameters.add(name, value = 0.0,min = min_external_parameters_value, max = max_external_parameters_value)


        mi = lmfit.minimize(self.calculateR,self.external_shape_parameters,args=(self.complete_array,fit_nbr),method='leastsq',maxfev = 10000)
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

    def external_shape_fitter_cartesian(self,verbose=True,show_graphic =True,interpolate =True):
        if not interpolate:
            self.external_shape_parameters = lmfit.Parameters()
            self.external_shape_parameters.add('A', value =1)
            self.external_shape_parameters.add('alpha',value = 1)
            self.external_shape_parameters.add('B',value = 1)
            self.external_shape_parameters.add('beta',value = 1)
            self.external_shape_parameters.add('const',value = 1)
            self.external_shape_parameters.add('delay',value = 1)

            mi = lmfit.minimize(self.calculateR_cartesian,self.external_shape_parameters,args=(self.complete_array,False),method='leastsq',maxfev = 10000)
            self.v = self.external_shape_parameters.valuesdict()

            y = np.arange(0,np.amax(self.complete_array[:,1]),0.1)
            self.fitted_external_shape = np.ndarray((y.shape[0],2),dtype=y.dtype)
            self.fitted_external_shape[:,0] = y
            self.fitted_external_shape[:,1] = self.calculateR_cartesian(mi.params,y,eval_flag=True)
            print "success: " + str(mi.success)
            if show_graphic:
                plt.plot(self.complete_array[:,1],self.complete_array[:,0],"r--",label="Experimental")
                plt.plot(y,self.fitted_external_shape[:,1],"b-",label="Fit")
                plt.legend()
                plt.show()
                # plt.close()

            if mi.success:
                if verbose:
                    verbose_dict = {}
                    verbose_dict["message"] = mi.message
                    verbose_dict["residual"] = mi.residual
                    verbose_dict["Iteration_number"] = mi.nfev
                    lmfit.printfuncs.report_fit(self.external_shape_parameters)
            return mi.success
        else:
            x = self.complete_array[:,0]
            y = self.complete_array[:,1]

            s = inter.UnivariateSpline(y,x, s=100000000)
            ynew = np.arange(0,np.amax(y),1)
            xnew = s(ynew)
            self.fitted_external_shape = np.ndarray((ynew.shape[0],2),dtype=y.dtype)
            self.fitted_external_shape[:,0] = ynew
            self.fitted_external_shape[:,1] = xnew
            plt.plot(y,x,"ro",label="Experimental")
            plt.plot(ynew[::-1],xnew[::-1],"b-",label="Fit")
            plt.legend()
            plt.show()
            return True

    def external_shape_fit(self,STEP=np.pi/1000,fit_nbr=1,x0=None,cartesian= False):
        if not cartesian:
            if x0 is None:
                w = self.thresholds['crop_w']
                l = (w-np.amin(self.complete_array[:,0]))/2.
                x0 = np.amin(self.complete_array[:,0])+l,0

            success = self.external_shape_fitter(x0,verbose=True,theta_free=True,fit_nbr=fit_nbr)
            if self.concavity_flag:
                if self.v["a2"]<0 or self.v['a3']<0:
                    success=False
            if success:
                theta_external = np.arange(0,2*np.pi , STEP)
                self.fitted_external_shape = self.generate_fitted_contour(self.v, fit_nbr, theta=theta_external)
                self.draw_contour(255, 1, contour=self.fitted_external_shape)
        else:
            success = self.external_shape_fitter_cartesian(verbose=True,show_graphic=True)
            if success:
                self.draw_contour(255, 1, contour=self.fitted_external_shape)

        return success

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
            theta_th = (np.cos(theta-v['theta0']))**i
            deltaR = deltaR * theta_th
            theoretical_radius  += deltaR
        theoretical_radius  = v['b0']* theoretical_radius
        return (experimental_radius-theoretical_radius)
    def calculateR_cartesian(self,parameters,data,eval_flag=False):
        if not eval_flag:
            v = parameters.valuesdict()
            X = data[:,1]
            Y = data[:,0]
            theoretical_X = 0
            theoretical_X  = v['A']*(Y-v['delay'])**(-v['alpha'])+v['B']*(Y-v['delay'])**(-v['beta'])+v['const']
            return (X-theoretical_X)
        else:
            v = parameters.valuesdict()
            Y = data
            theoretical_X  = v['A']*(Y-v['delay'])**(-v['alpha'])+v['B']*(Y-v['delay'])**(-v['beta'])+v['const']
            return theoretical_X


    def generate_fitted_contour(self, parameters, nbr, step=np.pi/1800.0, theta=None):
        if theta is None:
            theta = np.arange(-np.pi, np.pi, step)
        fitted_contour = 0
        derivate = 0
        for i in xrange (0,nbr+1):
            name = 'a'+str(i)
            # fitted_contour += parameters[name]*(np.sin(theta-parameters['theta0']))**i
            fitted_contour += parameters[name]*(np.cos(theta))**i
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
        output_array = np.ndarray((2,X.shape[0]),dtype=X.dtype)
        output_array[0] = Y
        output_array[1] = X
        output_array = output_array.transpose()
        return  output_array

    def draw_contour(self,color,thickness,contour=None,):
        """Draw the contour obtained by the ball_shape_fit"""

        Xexp = self.complete_array[:,0]
        Yexp = self.complete_array[:,1]
        # for i in xrange(Xexp.shape[0]):
        #     # cv2.line(self.in_img,(Xexp[i].astype(int),Yexp[i].astype(int)),(Xexp[i].astype(int),Yexp[i].astype(int)),color,thickness)
        Xexp = contour[:,1]
        Yexp = contour[:,0]
        for i in xrange(Xexp.shape[0]):
            # cv2.line(self.in_img,(Xexp[i].astype(int),Yexp[i].astype(int)),(Xexp[i].astype(int),Yexp[i].astype(int)),color,thickness)
            cv2.line(self.out_img,(Xexp[i].astype(int),Yexp[i].astype(int)),(Xexp[i].astype(int),Yexp[i].astype(int)),color,thickness)

    def rotate_contour(self,contour,x0,y0,theta0):
        # plt.plot(contour[:,0],contour[:,1],'.')
        # plt.show()
        rotated_contour = np.zeros(contour.shape,contour.dtype)
        contour[:,0] -= x0
        contour[:,1] -= y0
        rotated_contour[:, 0] = contour[:,0]*np.cos(theta0)+contour[:,1]*np.sin(theta0)
        rotated_contour[:, 1] = -contour[:,0]*np.sin(theta0)+contour[:,1]*np.cos(theta0)
        # plt.plot(rotated_contour[:,0],rotated_contour[:,1],'*')
        # plt.show()
        return rotated_contour

    def crop_img(self,in_img):
        """To be used in the beginning to reduce the size of the image to be treated"""
        x0 = self.thresholds["crop_x0"]
        y0 = self.thresholds["crop_y0"]
        h = self.thresholds["crop_h"]
        w = self.thresholds["crop_w"]
        out_img = in_img[y0:y0+h,x0:x0+w].copy()
        return out_img
if __name__ == "__main__":
    input_name= 'Input/Buckling_zoom_FPS_10k_00001'