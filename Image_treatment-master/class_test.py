__author__ = 'Adel'
import shape as sh

import numpy as np
import shutil
from manip_analyser import *

if __name__=="__main__":
    path_in = "glycerol/5_25/image_treatment_parameters_manip_1"
    image_in_name = 'Input/Buckling_1_1520.bmp'
    image_out_name ='Output/Buckling_1_1520.bmp'
    save_flag= True

    config_parameters = config_reader(path_in)
    prefix = 'buckling_'

    epsilon_coef = float(config_parameters[prefix+'epsilon'])
    nbr_p = config_parameters['fit_parameters_nbr']
    listExternalVolume = []
    listGravityCenter =  []
    listInternalVolume = []
    listLowPoint = []
    listExternalParameters = []
    listConcavityParameters = []
    listDeltaPressure = []
    listThetaConcavity = []
    canny_a = config_parameters['b_isotropic_canny_a']
    canny_b = config_parameters['b_isotropic_canny_b']
    temp_shape = shape_initializer(image_in_name,image_out_name,config_parameters,canny_a,canny_b)
    canny_a = config_parameters[prefix+'canny_a']
    canny_b = config_parameters[prefix+'canny_b']
    concavity_flag = False
    success_1=temp_shape.external_shape_fit(concavity_flag=concavity_flag,)
    concavity_flag = True
    if concavity_flag:
        theta_min,theta_max = temp_shape.calculate_fit_angle(analytical=False)
        epsilon = np.pi/epsilon_coef
        print "epsilon is: "+ str(epsilon)
        theta = temp_shape.tangent_minimal(theta_min,theta_max,epsilon = epsilon)

        phi0 = (np.pi/2.)-theta
        xc, yc = temp_shape.fit_linker(theta)
        tan_alpha = temp_shape.calculate_tangent_alpha(epsilon)
    else:
        theta = None
        phi0 = 0
        xc, yc = (0,0)
        tan_alpha = 0
    success_2 = temp_shape.ball_shape_fit(concavity_flag=concavity_flag, x_c=xc, y_c=yc, tan_alpha=tan_alpha, root_theta=theta, a=canny_a, b=canny_b, show_flag=True,
                                          clahe=True)


    if success_1 and success_2:
        if save_flag:
            dot = temp_shape.gravity_center(concavity_flag=concavity_flag,pix = True,phi0=phi0,R0=xc)
            temp_shape.draw_dot((0,0,0),3,dot)
            temp_shape.save(image_out_name)
        if concavity_flag:
            listThetaConcavity.append(theta)
            listConcavityParameters.append(temp_shape.c)
        else:
            listThetaConcavity.append(None)
            listConcavityParameters.append(None)

