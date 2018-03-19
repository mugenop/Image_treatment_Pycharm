import sys

import lmfit

__author__ = 'Adel'
import shape as sh
import os
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue

"""
TODO:

-Create one time array with a step of 1 s until the last image stack (where the buckling happens) where the step changes to 0.01s.
-Create one pressure array with a step of 1mbar.
-Create one stack of images matching exactly the time array (get rid of the images after 10s when there is no buckling) and select an image out of 100 during the 10s lap.

"""


def calculateConversion(params,shape,R):
    v = params.valuesdict()
    external_volume_th = (4*np.pi*R**3)/3.
    shape.thresholds['conversion']=v['conversion']
    external_volume = shape.compute_external_volume(concavity_flag=False, phi0=0, R0=0,)
    c =external_volume_th*np.ones(10)
    return external_volume_th*np.ones(10)-external_volume*np.ones(10)


def reference_pix_m(config_parameters,R=0.025):

    image_in_name = config_parameters['general_path_in']  + config_parameters['reference']
    image_out_name = None
    canny_a = config_parameters['b_isotropic_canny_a']
    canny_b = config_parameters['b_isotropic_canny_b']
    temp_shape = shape_initializer(image_in_name, image_out_name, config_parameters, canny_a, canny_b)
    temp_shape.draw_contour((0, 0, 0), 1, drawBallContour=True)
    temp_shape.external_shape_fit(concavity_flag=False)
    temp_shape.ball_shape_fit(concavity_flag=False)
    params = lmfit.Parameters()
    params.add('conversion', value =0.0001,vary = True )
    mi = lmfit.minimize(calculateConversion,params,args=(temp_shape,R),method='leastsq')
    v =params.valuesdict()
    d = v['conversion']
    config_parameters['conversion']=d
    external_volume = temp_shape.compute_external_volume(concavity_flag=False, phi0=0, R0=0)

    return d



def config_reader(path_in):
    f = open(path_in, 'r')
    lines = f.readlines()
    lines.remove("\n")
    threshold_dict = {}
    for el in lines:
        if "=" in el:
            try:
                el = el.rstrip('\n')
                parameter = el.replace(" ", "").split("=")
                if parameter[0].startswith("_"):
                    parameter = el.replace(" ", "").split("=")
                    threshold_dict[parameter[0][1:]] = parameter[1]
                elif parameter[1].startswith("0"):
                    parameter = el.replace(" ", "").split("=")
                    threshold_dict[parameter[0]] = float(parameter[1])

                else:
                    parameter = el.replace(" ", "").split("=")
                    threshold_dict[parameter[0]] = int(parameter[1])
            except ValueError:
                print el
        else:
            continue
    f.close()
    return threshold_dict


def array_constructor(array_string_config, fixed_pressure_array_shape=None, boolean=False):
    if boolean is not True:
        root_layer = array_string_config.split(",")
        list_layer = []
        for i in xrange(len(root_layer)):
            layer = root_layer[i].split("_")
            if len(layer) == 3:
                if float(layer[2]) > 0:
                    array = np.arange(float(layer[0]), float(layer[1]), float(layer[2]))
                else:
                    inversed_array = np.arange(float(layer[0]), float(layer[1]), abs(float(layer[2])))
                    array = inversed_array[::-1]
            else:
                if len(layer) == 1:
                    array = float(layer[0]) * np.ones(fixed_pressure_array_shape)
                else:
                    array = float(layer[0]) * np.ones((float(layer[1]),))
            list_layer.append(array)

        final_array = list_layer[0]
        print final_array.shape
        for i in xrange(1, len(list_layer)):
            new = list_layer[i]
            print new.shape
            final_array = np.concatenate((final_array, new), axis=0)

    else:
        root_layer = array_string_config.split("_")
        false_array = np.zeros(abs(float(root_layer[1]) - float(root_layer[0])), dtype=bool)
        true_array = np.ones(abs(float(root_layer[2]) - float(root_layer[1])), dtype=bool)
        if int(root_layer[0] < root_layer[-1]):
            final_array = np.concatenate((false_array, true_array), axis=0)
        else:
            final_array = np.concatenate((true_array, false_array), axis=0)

    return final_array


def file_header(wfile, results=False, external=False, concavity=False, error=False, nbrParameters=0):
    if results:
        wfile.writelines(
            "time\tpressure\timage_name\tdeltaPressure\tYlow\tGravity_X\tGravity_Y\tExternal_volume\tInternal_volume\tTheta_concavity\t")
    elif external:
        wfile.writelines("time\tpressure\timage_name\tb0\tx0\ty0\ttheta0\t")
        for i in xrange(nbrParameters + 1):
            name = "a" + str(i)
            wfile.writelines(name + '\t')
    elif concavity:
        wfile.writelines("time\tpressure\timage_name\txc\tyc\ttan_alpha\ta6\ta4\ta0\ta2")
    elif error:
        wfile.writelines("time\tpressure\timage_name\timage_name\tProblem\t")


def file_construct(path_out, experiment_name, concavity=False, nbr=0):
    if concavity:
        f_externalparameters = open(path_out + experiment_name + "external_fit_parameters.txt", "w")
        f_concavityparameters = open(path_out + experiment_name + "concavity_fit_parameters.txt", "w")
        f_results = open(path_out + experiment_name + "results.txt", "w")
        f_errors = open(path_out + experiment_name + "error.txt", "w")
        file_header(f_externalparameters, external=True, nbrParameters=nbr)
        file_header(f_concavityparameters, concavity=True)
        file_header(f_results, results=True)
        file_header(f_errors, error=True)
        f_externalparameters.close()
        f_concavityparameters.close()
        f_results.close()
        f_errors.close()

    else:
        f_externalparameters = open(path_out + experiment_name + "external_fit_parameters.txt", "w")
        f_results = open(path_out + experiment_name + "results.txt", "w")
        f_errors = open(path_out + experiment_name + "error.txt", "w")
        file_header(f_externalparameters, external=True, nbrParameters=nbr)
        file_header(f_results, results=True)
        file_header(f_errors, error=True)
        f_externalparameters.close()
        f_results.close()
        f_errors.close()


def parameters_writer(wfile, t, p, name, lp):
    wfile.writelines("\n")
    wfile.writelines("{0:.5e}".format(t) + '\t')
    wfile.writelines("{0:.5e}".format(p) + '\t')
    wfile.writelines(name + '\t')
    if lp is not None:
        for val in lp:
            wfile.writelines("{0:.3e}".format(lp[val]) + "\t")
    else:
        wfile.writelines("N\A" + '\t')


def results_writer(wfile, t, p, name, deltap, Ylow, gravity_center, external_volume, internal_volume,
                   theta_concavity=None):
    if theta_concavity is not None:
        wfile.writelines("\n")
        wfile.writelines("{0:.5e}".format(t) + '\t')
        wfile.writelines("{0:.5e}".format(p) + '\t')
        wfile.writelines(name + '\t')
        wfile.writelines("{0:.5e}".format(deltap) + '\t')
        wfile.writelines("{0:.5e}".format(Ylow) + '\t')
        wfile.writelines("{0:.5e}".format(gravity_center[0]) + '\t')
        wfile.writelines("{0:.5e}".format(gravity_center[1]) + '\t')
        wfile.writelines("{0:.5e}".format(external_volume) + '\t')
        wfile.writelines("{0:.5e}".format(internal_volume) + '\t')
        wfile.writelines("{0:.5e}".format(theta_concavity) + '\t')
    else:

        wfile.writelines("\n")
        wfile.writelines("{0:.5e}".format(t) + '\t')
        wfile.writelines("{0:.5e}".format(p) + '\t')
        wfile.writelines(name + '\t')
        wfile.writelines("{0:.5e}".format(deltap) + '\t')
        wfile.writelines("{0:.5e}".format(Ylow) + '\t')
        wfile.writelines("{0:.5e}".format(gravity_center[0]) + '\t')
        wfile.writelines("{0:.5e}".format(gravity_center[1]) + '\t')
        wfile.writelines("{0:.5e}".format(external_volume) + '\t')
        wfile.writelines("{0:.5e}".format(internal_volume) + '\t')
        wfile.writelines('N\A' + '\t')


def array_writer(wfile, dp, dv):
    wfile.writelines("deltaPressure\tdelta_volume\t")
    for i in xrange(dp.shape[0]):
        wfile.writelines("\n")
        wfile.writelines("{0:.5e}".format(dp[i]) + '\t')
        wfile.writelines("{0:.5e}".format(dv[i]) + '\t')


def isotropic(config_parameters, prefix, save_flag=True, show_Flag_canny=False):
    time_array_config = config_parameters[prefix + 'time']
    pressure_array_config = config_parameters[prefix + 'pressure']
    path_in_general = config_parameters['general_path_in']
    path_out_general = config_parameters['general_path_out']
    path_in = config_parameters[prefix + 'path_in']
    path_out = config_parameters[prefix + 'path_out']
    image_out = config_parameters[prefix + 'image_out']
    format_out = config_parameters[prefix + 'format_out']
    time_array = array_constructor(time_array_config)
    pressure_array = array_constructor(pressure_array_config) * 100 + config_parameters['internal_pressure']

    print "time array:"
    print time_array
    print "Pressure array:"
    print pressure_array

    if time_array.shape[0] != pressure_array.shape[0]:

        print "Size error time and pressure"
        print time_array.shape[0]
        print pressure_array.shape[0]
        return None
    else:
        experiment_name = config_parameters['manip_name'] + prefix
        listExternalVolume = []
        listGravityCenter = []
        listInternalVolume = []
        listLowPoint = []
        listExternalParameters = []
        listDeltaPressure = []
        path_in_images = path_in_general + path_in
        path_out_images = path_out_general + path_out
        if not os.path.isdir(path_out_images):
            os.makedirs(path_out_images)
        file_construct(path_out_images, experiment_name=experiment_name)
        if pressure_array[0] < pressure_array[-1]:
            images_list = os.listdir(path_in_images)
        else:
            images_list = os.listdir(path_in_images)
            images_list.reverse()
        for i in xrange(len(images_list)):
            print 'experiment: '+ config_parameters['manip_name']+'\ttreating image ' + str(i) + ' over ' + str(len(images_list))
            suffix = str((pressure_array[i] - config_parameters['internal_pressure']) / 100.)
            image_in_name = path_in_images + images_list[i]
            image_out_name = path_out_images + image_out + suffix + format_out
            canny_a = config_parameters['b_isotropic_canny_a']
            canny_b = config_parameters['b_isotropic_canny_b']
            temp_shape = shape_initializer(image_in_name, image_out_name, config_parameters, canny_a, canny_b,
                                           show_flag_canny=show_Flag_canny)
            temp_shape.draw_contour((0, 0, 0), 1, drawBallContour=True)
            success_1 = temp_shape.external_shape_fit(concavity_flag=False, circular=True)
            temp_shape.ball_shape_fit(concavity_flag=False)
            if success_1:
                if save_flag:
                    dot = temp_shape.gravity_center(concavity_flag=False, pix=True)
                    temp_shape.draw_dot((0, 0, 0), 3, dot)
                    temp_shape.save(image_out_name)

                shape_postprocessing(temp_shape, pressure_array[i], False, listExternalParameters, listExternalVolume,
                                     listInternalVolume, listDeltaPressure, listGravityCenter, listLowPoint)
                save_data = True
                if len(listLowPoint) >= 2:
                    y_k = listLowPoint[-1]
                    y_k_1 = listLowPoint[-2]
                    if y_k - y_k_1 > 0.01:
                        f_errors = open(path_out_images + experiment_name + "error.txt", "a")
                        f_errors.writelines('\n')
                        f_errors.writelines(str(time_array[i]) + '\t' + str(pressure_array[i]) + '\t' + images_list[
                            i] + '\t' + 'gravity_center_fail\t')
                        f_errors.close()
                        save_data = False
                    extV_k = listExternalVolume[-1]
                    if abs(extV_k) > 7 * 10 ** -5:
                        f_errors = open(path_out_images + experiment_name + "error.txt", "a")
                        f_errors.writelines('\n')
                        f_errors.writelines(str(time_array[i]) + '\t' + str(pressure_array[i]) + '\t' + images_list[
                            i] + '\t' + 'Volume_fail\t')
                        f_errors.close()
                        save_data = False
                if save_data:
                    f_externalparameters = open(path_out_images + experiment_name + "external_fit_parameters.txt", "a")
                    # print "Volume is : "+str(listExternalVolume[0])
                    f_results = open(path_out_images + experiment_name + "results.txt", "a")
                    results_writer(f_results, time_array[i], pressure_array[i], images_list[i], listDeltaPressure[-1],
                                   listLowPoint[-1], listGravityCenter[-1], listExternalVolume[-1],
                                   listInternalVolume[-1])
                    parameters_writer(f_externalparameters, time_array[i], pressure_array[i], images_list[i],
                                      listExternalParameters[-1])
                    f_externalparameters.close()
                    f_results.close()

            else:
                f_errors = open(path_out_images + experiment_name + "error.txt", "a")
                f_errors.writelines('\n')
                f_errors.writelines(
                    str(time_array[i]) + '\t' + str(pressure_array[i]) + '\t' + images_list[i] + '\t' + 'fit_fail\t')
                f_errors.close()
        V0 = config_parameters['internal_volume'] + 0.5 * config_parameters['shell_volume']
        deltaV = 1 - ((np.array(listExternalVolume) - 0.5 * config_parameters['shell_volume']) / V0)
        deltaP = np.array(listDeltaPressure)

        return deltaP, deltaV


def rolling(config_parameters, save_flag=True, rolling_dynamic=True):
    prefix = 'db_rolling_'
    time_array_config = config_parameters[prefix + 'time']
    pressure_array_config = config_parameters[prefix + 'pressure']
    path_in_general = config_parameters['general_path_in']
    path_out_general = config_parameters['general_path_out']
    path_in = config_parameters[prefix + 'path_in']
    path_out = config_parameters[prefix + 'path_out']
    image_in = config_parameters[prefix + 'image_in']
    image_out = config_parameters[prefix + 'image_out']
    format_in = config_parameters[prefix + 'format_in']
    format_out = config_parameters[prefix + 'format_out']
    time_array = array_constructor(time_array_config)
    pressure_array = array_constructor(pressure_array_config) * 100 + config_parameters['internal_pressure']
    print "time array:"
    print time_array
    print "Pressure array:"
    print pressure_array
    if time_array.shape[0] != pressure_array.shape[0]:
        print "Size error time and pressure"
        print time_array.shape[0]
        print pressure_array.shape[0]
        return None
    else:
        canny_a = config_parameters[prefix + 'canny_a']
        canny_b = config_parameters[prefix + 'canny_b']
        canny_a_iso = config_parameters['b_isotropic_canny_a']
        canny_b_iso = config_parameters['b_isotropic_canny_b']
        epsilon_coef = float(config_parameters[prefix + 'epsilon'])
        nbr_p = config_parameters['fit_parameters_nbr']
        experiment_name = config_parameters['manip_name'] + prefix

        listExternalVolume = []
        listGravityCenter = []
        listInternalVolume = []
        listLowPoint = []
        listExternalParameters = []
        listConcavityParameters = []
        listDeltaPressure = []
        listThetaConcavity = []
        path_in_images = path_in_general + path_in
        path_out_images = path_out_general + path_out
        if not os.path.isdir(path_out_images):
            os.makedirs(path_out_images)
        if pressure_array[0] <= pressure_array[-1] or rolling_dynamic:
            images_list = os.listdir(path_in_images)
        else:
            images_list = os.listdir(path_in_images)
            images_list.reverse()

        file_construct(path_out_images, experiment_name=experiment_name, concavity=True, nbr=nbr_p)
        beg = 0
        for i in xrange(beg, len(images_list)):
            print 'experiment: '+ config_parameters['manip_name']+'\ttreating image ' + str(i) + ' over ' + str(len(images_list))
            suffix = str((pressure_array[i] - config_parameters['internal_pressure']) / 100.)
            image_in_name = path_in_images + images_list[i]
            image_out_name = path_out_images + images_list[i]
            temp_shape = shape_initializer(image_in_name, image_out_name, config_parameters, canny_a_iso, canny_b_iso)
            success_1 = temp_shape.external_shape_fit(concavity_flag=True)
            theta_min, theta_max = temp_shape.calculate_fit_angle(analytical=False)
            epsilon = np.pi / epsilon_coef
            theta = temp_shape.tangent_minimal(theta_min, theta_max, epsilon=epsilon)
            phi0 = (np.pi / 2.) - theta
            xc, yc = temp_shape.fit_linker(theta)
            tan_alpha = temp_shape.calculate_tangent_alpha(theta)
            success_2 = temp_shape.ball_shape_fit(concavity_flag=True, x_c=xc, y_c=yc, tan_alpha=tan_alpha,
                                                  root_theta=theta, a=canny_a, b=canny_b, clahe=False)
            if success_1 and success_2:
                if save_flag:
                    dot = temp_shape.gravity_center(concavity_flag=True, pix=True, phi0=phi0, R0=xc)
                    temp_shape.draw_dot((0, 0, 0), 3, dot)
                    temp_shape.save(image_out_name)
                shape_postprocessing(temp_shape, pressure_array[i], True, listExternalParameters, listExternalVolume,
                                     listInternalVolume, listDeltaPressure, listGravityCenter, listLowPoint, phi0=phi0,
                                     xc=xc)
                save_data = True
                if len(listLowPoint) >= 2:
                    y_k = listLowPoint[-1]
                    y_k_1 = listLowPoint[-2]
                    if y_k - y_k_1 > 0.01:
                        f_errors = open(path_out_images + experiment_name + "error.txt", "a")
                        f_errors.writelines('\n')
                        f_errors.writelines(str(time_array[i]) + '\t' + str(pressure_array[i]) + '\t' + images_list[
                            i] + '\t' + 'gravity_center_fail\t')
                        f_errors.close()
                        save_data = False
                    extV_k = listExternalVolume[-1]
                    if abs(extV_k) > 7 * 10 ** -5:
                        f_errors = open(path_out_images + experiment_name + "error.txt", "a")
                        f_errors.writelines('\n')
                        f_errors.writelines(str(time_array[i]) + '\t' + str(pressure_array[i]) + '\t' + images_list[
                            i] + '\t' + 'Volume_fail\t')
                        f_errors.close()
                        save_data = False
                if save_data:
                    listConcavityParameters.append(temp_shape.c)
                    listThetaConcavity.append(theta)
                    f_externalparameters = open(path_out_images + experiment_name + "external_fit_parameters.txt", "a")
                    f_concavityparameters = open(path_out_images + experiment_name + "concavity_fit_parameters.txt",
                                                 "a")
                    f_results = open(path_out_images + experiment_name + "results.txt", "a")
                    results_writer(f_results, time_array[i], pressure_array[i], images_list[i], listDeltaPressure[-1],
                                   listLowPoint[-1], listGravityCenter[-1], listExternalVolume[-1],
                                   listInternalVolume[-1], listThetaConcavity[-1])
                    parameters_writer(f_externalparameters, time_array[i], pressure_array[i], images_list[i],
                                      listExternalParameters[-1])
                    parameters_writer(f_concavityparameters, time_array[i], pressure_array[i], images_list[i],
                                      listConcavityParameters[-1])
                    f_externalparameters.close()
                    f_concavityparameters.close()
                    f_results.close()
            else:
                f_errors = open(path_out_images + experiment_name + "error.txt", "a")
                f_errors.writelines('\n')
                f_errors.writelines(str(time_array[i]) + '\t' + str(pressure_array[i]) + '\t' + images_list[i] + '\t')
                f_errors.close()

        V0 = config_parameters['internal_volume'] + 0.5 * config_parameters['shell_volume']
        deltaV = 1 - ((np.array(listExternalVolume) - 0.5 * config_parameters['shell_volume']) / V0)
        deltaP = np.array(listDeltaPressure)
        return deltaP, deltaV


def dynamic(config_parameters, prefix, save_flag=True, i_image=0):
    time_array_config = config_parameters[prefix + 'time']
    pressure_array_config = config_parameters[prefix + 'pressure']
    cavity_array_config = config_parameters[prefix + 'concavity']
    path_in_general = config_parameters['general_path_in']
    path_out_general = config_parameters['general_path_out']
    path_in = config_parameters[prefix + 'path_in']
    path_out = config_parameters[prefix + 'path_out']
    image_in = config_parameters[prefix + 'image_in']
    image_out = config_parameters[prefix + 'image_out']
    format_in = config_parameters[prefix + 'format_in']
    format_out = config_parameters[prefix + 'format_out']
    time_array = array_constructor(time_array_config)
    print "time array:"
    print time_array
    pressure_array = array_constructor(pressure_array_config, fixed_pressure_array_shape=time_array.shape) * 100 + \
                     config_parameters['internal_pressure']
    print "Pressure array:"
    print pressure_array
    concavity_array = array_constructor(cavity_array_config, boolean=True)
    if (time_array.shape[0] != pressure_array.shape[0]) or (concavity_array.shape[0] != pressure_array.shape[0]):
        print "Size error time and pressure"
        print time_array.shape[0]
        print pressure_array.shape[0]
        print concavity_array.shape[0]
        return None
    else:

        canny_a = config_parameters[prefix + 'canny_a']
        canny_b = config_parameters[prefix + 'canny_b']
        canny_a_iso = config_parameters['b_isotropic_canny_a']
        canny_b_iso = config_parameters['b_isotropic_canny_b']
        epsilon_coef = float(config_parameters[prefix + 'epsilon'])
        nbr_p = config_parameters['fit_parameters_nbr']
        experiment_name = config_parameters['manip_name'] + prefix

        listExternalVolume = []
        listGravityCenter = []
        listInternalVolume = []
        listLowPoint = []
        listExternalParameters = []
        listConcavityParameters = []
        listDeltaPressure = []
        listThetaConcavity = []
        path_in_images = path_in_general + path_in
        path_out_images = path_out_general + path_out
        if not os.path.isdir(path_out_images):
            os.makedirs(path_out_images)
        if pressure_array[0] <= pressure_array[-1]:
            images_list = os.listdir(path_in_images)
        else:
            images_list = os.listdir(path_in_images)
            images_list.reverse()

        file_construct(path_out_images, experiment_name=experiment_name, concavity=True, nbr=nbr_p)

        i = i_image
        while i < len(images_list):
            error_msg="Before fits"
            print 'experiment: '+ config_parameters['manip_name']+'\ttreating image ' + str(i) + ' over ' + str(len(images_list))
            concavity_flag = concavity_array[i]
            image_in_name = path_in_images + images_list[i]
            image_out_name = path_out_images + images_list[i]
            success_1 =False
            temp_shape = shape_initializer(image_in_name, image_out_name, config_parameters, canny_a_iso, canny_b_iso)
            try:

                success_1 = temp_shape.external_shape_fit(concavity_flag=concavity_flag, )
            except:
                success_1 = False
                exc_tuple = sys.exc_info()
                error_msg =str(exc_tuple[0]) + str(exc_tuple[1])

            if success_1:
                if concavity_flag:
                    theta_min, theta_max = temp_shape.calculate_fit_angle(analytical=False)
                    epsilon = np.pi / epsilon_coef
                    print "epsilon is: " + str(epsilon)
                    theta = temp_shape.tangent_minimal(theta_min, theta_max, epsilon=epsilon)

                    phi0 = (np.pi / 2.) - theta
                    xc, yc = temp_shape.fit_linker(theta)
                    tan_alpha = temp_shape.calculate_tangent_alpha(epsilon)
                else:
                    theta = None
                    phi0 = 0
                    xc, yc = (0, 0)
                    tan_alpha = 0
                try:

                    success_2 = temp_shape.ball_shape_fit(concavity_flag=concavity_flag, x_c=xc, y_c=yc,
                                                          tan_alpha=tan_alpha, root_theta=theta, a=canny_a, b=canny_b,
                                                          clahe=False,show_flag=True)
                except:
                    success_2 = False
                    exc_tuple = sys.exc_info()
                    error_msg = str(exc_tuple[0]) + str(exc_tuple[1])
                if success_1 and success_2:
                    if save_flag:
                        dot = temp_shape.gravity_center(concavity_flag=concavity_flag, pix=True, phi0=phi0, R0=xc)
                        temp_shape.draw_dot((0, 0, 0), 3, dot)
                        print image_out_name
                        temp_shape.save(image_out_name)
                        temp_shape.image_show()
                    if concavity_flag:
                        listThetaConcavity.append(theta)
                        listConcavityParameters.append(temp_shape.c)
                    else:
                        listThetaConcavity.append(None)
                        listConcavityParameters.append(None)
                    shape_postprocessing(temp_shape, pressure_array[i], concavity_flag, listExternalParameters,
                                         listExternalVolume, listInternalVolume, listDeltaPressure, listGravityCenter,
                                         listLowPoint, phi0=phi0, xc=xc)
                    save_data = True
                    if len(listLowPoint) >= 2:
                        y_k = listLowPoint[-1]
                        y_k_1 = listLowPoint[-2]
                        if y_k - y_k_1 > 0.01:
                            error_msg= 'gravity_center_fail'
                            save_data = False
                        extV_k = listExternalVolume[-1]
                        if abs(extV_k) > 7 * 10 ** -5:
                            error_msg = 'Volume_fail'
                            save_data = False
                    if save_data:
                        f_externalparameters = open(path_out_images + experiment_name + "external_fit_parameters.txt", "a")
                        f_concavityparameters = open(path_out_images + experiment_name + "concavity_fit_parameters.txt",
                                                     "a")
                        f_results = open(path_out_images + experiment_name + "results.txt", "a")
                        results_writer(f_results, time_array[i], pressure_array[i], images_list[i], listDeltaPressure[-1],
                                       listLowPoint[-1], listGravityCenter[-1], listExternalVolume[-1],
                                       listInternalVolume[-1], listThetaConcavity[-1])
                        parameters_writer(f_externalparameters, time_array[i], pressure_array[i], images_list[i],
                                          listExternalParameters[-1])
                        parameters_writer(f_concavityparameters, time_array[i], pressure_array[i], images_list[i],
                                          listConcavityParameters[-1])
                        f_externalparameters.close()
                        f_concavityparameters.close()
                        f_results.close()
                else:

                    # if not success_2:
                    #     error_msg= "Concavity fit error"
                    f_errors = open(path_out_images + experiment_name + "error.txt", "a")
                    f_errors.writelines('\n')
                    f_errors.writelines(str(time_array[i]) + '\t' + str(pressure_array[i]) + '\t' + images_list[i] + '\t'+error_msg+'\t')
                    f_errors.close()
            else:
                    if not success_1:
                        error_msg = "External fit error"
                    f_errors = open(path_out_images + experiment_name + "error.txt", "a")
                    f_errors.writelines('\n')
                    f_errors.writelines(str(time_array[i]) + '\t' + str(pressure_array[i]) + '\t' + images_list[i] + '\t'+error_msg+'\t')
                    f_errors.close()

            i += 1

        V0 = config_parameters['internal_volume'] + 0.5 * config_parameters['shell_volume']
        deltaV = 1 - ((np.array(listExternalVolume) - 0.5 * config_parameters['shell_volume']) / V0)
        deltaP = np.array(listDeltaPressure)
        return deltaP, deltaV



def shape_initializer(path_in, path_out, thresholds, canny_a, canny_b, show_flag_canny=True,flag_rotation=False):
    # if thresholds["conversion"]==0:

    shape = sh.Shape(path_in, path_out=path_out, thresholds=thresholds)
    shape.crop_img()
    if flag_rotation:
        shape.rotate_image()
    shape.median_blur()
    shape.gaussian_blur()
    shape.find_general_shape(canny_a, canny_b, show_flag_canny)
    shape.find_ball_shape()
    return shape


def shape_postprocessing(temp_shape, p, concavity_flag, lep, lev, liv, ldp, lgc, llp, phi0=0, xc=0):
    external_volume = temp_shape.compute_external_volume(concavity_flag=concavity_flag, phi0=phi0, R0=xc)
    volume_shell = temp_shape.thresholds['shell_volume']
    internal_volume = external_volume - volume_shell
    internal_pressure = temp_shape.thresholds['internal_volume'] * temp_shape.thresholds[
        'internal_pressure'] / internal_volume
    deltaPressure = p - internal_pressure
    lep.append(temp_shape.v)
    lev.append(external_volume)
    liv.append(internal_volume)
    ldp.append(deltaPressure)
    lgc.append(temp_shape.gravity_center(concavity_flag=concavity_flag, phi0=phi0, R0=xc))
    llp.append(temp_shape.get_down_point())


def images_analyser(path_in, b_isotropic=(True, True, True), buckling=(True, True), roll=(True, True, False),
                    debuckling=(True, True), db_isotropic=(True, True), show_flag=True, queue=None):
    deltaP = None
    deltaV = None
    config_parameters = config_reader(path_in)
    config_parameters["conversion"] = reference_pix_m(config_parameters)
    if not os.path.isdir(config_parameters['general_path_out']):
        os.makedirs(config_parameters['general_path_out'])
    print "treating experiment: " + config_parameters['manip_name']
    if b_isotropic[0]:
        prefix = 'b_isotropic_'
        print prefix + " in progress"
        deltaP, deltaV = isotropic(config_parameters, prefix=prefix, save_flag=b_isotropic[1],
                                   show_Flag_canny=b_isotropic[2])
        print prefix + " Done"

    if buckling[0]:
        prefix = 'buckling_'
        print prefix + " in progress"
        if len(buckling) > 2:
            deltaP_temp, deltaV_temp = dynamic(config_parameters, prefix=prefix, save_flag=buckling[1],i_image=buckling[2])
        else:
            deltaP_temp, deltaV_temp = dynamic(config_parameters, prefix=prefix, save_flag=buckling[1], i_image=0)
        if deltaP is not None:
            deltaP = np.concatenate((deltaP, deltaP_temp), axis=0)
            deltaV = np.concatenate((deltaV, deltaV_temp), axis=0)
        else:
            deltaP = deltaP_temp
            deltaV = deltaV_temp
        print prefix + " Done"
    #
    if roll[0]:
        prefix = 'db_rolling_'
        print prefix + " in progress"
        deltaP_temp, deltaV_temp = rolling(config_parameters, save_flag=roll[1], rolling_dynamic=roll[2])
        if deltaP is not None:
            deltaP = np.concatenate((deltaP, deltaP_temp), axis=0)
            deltaV = np.concatenate((deltaV, deltaV_temp), axis=0)
        else:
            deltaP = deltaP_temp
            deltaV = deltaV_temp

    if debuckling[0]:
        prefix = 'debuckling_'
        print prefix + " in progress"
        deltaP_temp, deltaV_temp = dynamic(config_parameters, prefix, save_flag=debuckling[1])
        if deltaP is not None:
            deltaP = np.concatenate((deltaP, deltaP_temp), axis=0)
            deltaV = np.concatenate((deltaV, deltaV_temp), axis=0)
        else:
            deltaP = deltaP_temp
            deltaV = deltaV_temp
        print prefix + " Done"

    if db_isotropic[0]:

        prefix = 'db_isotropic_'
        print prefix + " in progress"
        deltaP_temp, deltaV_temp = isotropic(config_parameters, prefix=prefix, save_flag=db_isotropic[1])
        if deltaP is not None:
            deltaP = np.concatenate((deltaP, deltaP_temp), axis=0)
            deltaV = np.concatenate((deltaV, deltaV_temp), axis=0)
        else:
            deltaP = deltaP_temp
            deltaV = deltaV_temp
        print prefix + " Done"
    if deltaP is not None:
        plt.savefig(config_parameters['general_path_out'] + "dpdv.png")
        fdata = open(config_parameters['general_path_out'] + "dpdv.txt", "w")
        array_writer(fdata, deltaP, deltaV)
        plt.plot(deltaV, deltaP, label="dp= f(dv)")
        plt.legend()
        plt.show()
    else:
        print "Nothing to record"
    if queue is not None:
        queue.put(path_in + " is done.")


def multiple_image_analyser(title, number,EQUILIBRIUM=False,free_oscillation=False):
    queue_list = []
    process_list = []
    if EQUILIBRIUM:
        b_iso = False,False,False
        buck = True,True,0
        rol = False,False,False
        debuck = True,True
        db_iso = False,False
    if free_oscillation:
        b_iso = False,False,False
        buck = True,True,0
        rol = False,False,False
        debuck = False,True
        db_iso = False,False
    for i in xrange(number):
        queue_list.append(Queue())
    for i in xrange(number):
        if not EQUILIBRIUM and not free_oscillation:
            # if i==1:
            #     b_iso = True,True,True
            #     buck = False,True,0
            #     rol = True,True,False
            #     debuck = False,True
            #     db_iso = True,True
            if i==11111:
                b_iso = True,True,True
                buck = False,True,0
                rol = True,True,False
                debuck = False,True
                db_iso = True,True
            else:
                b_iso = False,True,True
                buck = False,True,0
                rol = False,True,False
                debuck = True,True
                db_iso = False,True
        titler = title + str(i+2)
        process = Process(target=images_analyser, kwargs={'path_in': titler, 'queue': queue_list[i],'buckling':buck,'b_isotropic':b_iso,'roll':rol,'debuckling':debuck,'db_isotropic':db_iso})
        process_list.append(process)
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()
    for q in queue_list:
        print q.get()

    print "processing..."


def main():
    multiple_image_analyser("Bruno/AJO_121/image_treatment_parameters_manip_",1)

if __name__ == '__main__':
    # path_in = "glycerol/2_25/image_treatment_parameters_manip_5"
    # b_iso = True,True,True
    # buck = True,True,0
    # #third component to handle the naming of rolling True if naming independant from pressure
    # rol = True,True,False
    # debuck = True,True
    # db_iso = True,True
    # # calibrate volumes first
    #
    # #ro
    # images_analyser(path_in,b_iso,buck,rol,debuck,db_iso)
    main()
