__author__ = 'Adel'
import shape as sh
import os
import numpy as np
from matplotlib import pyplot as plt


"""
TODO:

-Create one time array with a step of 1 s until the last image stack (where the buckling happens) where the step changes to 0.01s.
-Create one pressure array with a step of 1mbar.
-Create one stack of images matching exactly the time array (get rid of the images after 10s when there is no buckling) and select an image out of 100 during the 10s lap.

"""






def config_reader(path_in):
    f = open(path_in, 'r')
    lines=f.readlines()
    lines.remove("\n")
    threshold_dict = {}
    for el in lines:
        if "=" in el :
            el = el.rstrip('\n')
            parameter = el.replace(" ","").split("=")
            if parameter[0].startswith("_"):
                parameter = el.replace(" ","").split("=")
                threshold_dict[parameter[0][1:]] = parameter[1]
            elif parameter[1].startswith("0"):
                parameter = el.replace(" ","").split("=")
                threshold_dict[parameter[0]] = float(parameter[1])

            else:
                parameter = el.replace(" ","").split("=")
                threshold_dict[parameter[0]] = int(parameter[1])
        else:
            continue
    f.close()
    return threshold_dict

def shape_initializer(path_in,path_out,thresholds):
    shape = sh.Shape(path_in,path_out=path_out,thresholds=thresholds)
    shape.crop_img()
    shape.median_blur()
    shape.gaussian_blur()
    shape.find_general_shape(50,20)
    shape.find_ball_shape()
    return shape

def shape_postprocessing(temp_shape,p,concavity_flag,lep,lev,liv,ldp,lgc,llp,phi0=0,xc=0):
    external_volume = temp_shape.compute_external_volume(concavity_flag=concavity_flag,phi0=phi0,R0=xc)
    volume_shell = temp_shape.thresholds['shell_volume']
    internal_volume = external_volume-volume_shell
    internal_pressure = temp_shape.thresholds['internal_volume']*temp_shape.thresholds['internal_pressure']/internal_volume
    deltaPressure = p-internal_pressure
    lep.append(temp_shape.v)
    lev.append(external_volume)
    liv.append(internal_volume)
    ldp.append(deltaPressure)
    lgc.append(temp_shape.gravity_center(concavity_flag=concavity_flag,phi0=phi0,R0=xc))
    llp.append(temp_shape.get_down_point())

def images_analyser(path_in,b_isotropic=(True,True),buckling=(True,True),roll=(True,True),debuckling=(True,True),db_isotropic=(True,True)):
    deltaP = None
    deltaV = None
    config_parameters = config_reader(path_in)
    print "treating experiment: "+config_parameters['manip_name']
    if b_isotropic[0]:
        prefix ='b_isotropic_'
        print prefix+ " in progress"
        deltaP,deltaV=isotropic(config_parameters,prefix=prefix,save_flag=b_isotropic[1])
        print prefix+ " Done"

    if buckling[0]:
        prefix ='buckling_'
        print prefix+ " in progress"
        deltaP_temp,deltaV_temp=dynamic(config_parameters,prefix=prefix,save_flag=buckling[1])
        if deltaP is not None:
            deltaP = np.concatenate((deltaP,deltaP_temp),axis=0)
            deltaV = np.concatenate((deltaV,deltaV_temp),axis=0)
        else:
            deltaP = deltaP_temp
            deltaV = deltaV_temp
        print prefix+ " Done"
    #
    if roll[0]:
        prefix ='db_rolling_'
        print prefix+ " in progress"
        deltaP_temp,deltaV_temp = rolling(config_parameters,save_flag = roll[1],rolling_dynamic=roll[2])
        if deltaP is not None:
            deltaP = np.concatenate((deltaP,deltaP_temp),axis=0)
            deltaV = np.concatenate((deltaV,deltaV_temp),axis=0)
        else:
            deltaP = deltaP_temp
            deltaV = deltaV_temp

    if debuckling[0]:
        prefix ='debuckling_'
        print prefix+ " in progress"
        deltaP_temp,deltaV_temp=dynamic(config_parameters,prefix,save_flag=debuckling[1])
        if deltaP is not None:
            deltaP = np.concatenate((deltaP,deltaP_temp),axis=0)
            deltaV = np.concatenate((deltaV,deltaV_temp),axis=0)
        else:
            deltaP = deltaP_temp
            deltaV = deltaV_temp
        print prefix+ " Done"

    if db_isotropic[0]:

        prefix ='db_isotropic_'
        print prefix+ " in progress"
        deltaP_temp,deltaV_temp=isotropic(config_parameters,prefix=prefix,save_flag=db_isotropic[1])
        if deltaP is not None:
            deltaP = np.concatenate((deltaP,deltaP_temp),axis=0)
            deltaV = np.concatenate((deltaV,deltaV_temp),axis=0)
        else:
            deltaP = deltaP_temp
            deltaV = deltaV_temp
        print prefix+ " Done"
    if deltaP is not None:
        plt.savefig(config_parameters['general_path_out']+"dpdv.png")
        fdata = open(config_parameters['general_path_out']+"dpdv.txt","w")
        array_writer(fdata,deltaP,deltaV)
        plt.plot(deltaV,deltaP,label="dp= f(dv)")
        plt.legend()
        plt.show()
    else:
        print "Nothing to record"
if __name__=='__main__':

    path_in = "image_treatment_parameters_manip_2"
    b_iso = True,True
    buck = False,True
    #third component to handle the naming of rolling True if naming independant from pressure
    rol = False,True,False
    debuck = False,True
    db_iso = False,True
    # calibrate volumes first
    #ro
    images_analyser(path_in,b_iso,buck,rol,debuck,db_iso)
