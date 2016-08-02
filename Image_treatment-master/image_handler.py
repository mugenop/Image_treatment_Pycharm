__author__ = 'Adel'
import os
import shutil
import numpy as np
import cv2
from PIL import Image
PROCESS = False
RENAME = False
RENAMERAMP = True
CONVERT = False
BITMAP = False
def image_stack_processor(path_in,filename_in,path_out,filename_out,step,number_out,stop = None):
    """

    :param path_in: dossier du fichier ou aller chercher les images.
    :param filename_in: sert a spliter la chaine de caracteres dans les noms des images.
    :param path_out: dossier ou stocker les images renommees.
    :param filename_out: obvious
    :param step: si le nombre d'images ne correspond pas au nombre qu'on veut traiter, on applique un pas pour etre en accord avec le vecteur temps/pression.
    :param number_out: continuer le decompte des images.
    :param stop: nombre d'images a traiter dans un dossier.
    :return: number_out+nombre d'images mv/rename dans la fonction.

    """

    k = number_out
    i = '{0:05d}'.format(k)
    if stop == None:

        for filename in os.listdir(path_in):
                if filename.startswith(filename_in):
                    img_nbr =int(filename.split(filename_in)[1].split(".")[0])
                    if (img_nbr%step ==0):
                        shutil.copyfile(path_in+"\\"+filename, path_out+"\\"+filename_out+i+".bmp")
                        k+=1
                        i = '{0:05d}'.format(k)
                        print k
    else:
        for filename in os.listdir(path_in):
                if  k-number_out<stop:
                    if filename.startswith(filename_in):
                        img_nbr =int(filename.split(filename_in)[1].split(".")[0])
                        if (img_nbr%step ==0):
                            shutil.copyfile(path_in+"\\"+filename, path_out+"\\"+filename_out+i+".bmp")
                            k+=1
                            i = '{0:05d}'.format(k)
                            print k
                else:
                    break

    return k

def rename_image(path_in,filename_in,path_out,filename_out,step,number_init,extension=".bmp"):

    for filename in os.listdir(path_in):
            if filename.startswith(filename_in):
                img_nbr =int(filename.split(filename_in)[1].split(".")[0])
                img_nbr = number_init+img_nbr-1
                if (img_nbr%step ==0):
                    shutil.copyfile(path_in+"\\"+filename, path_out+"\\"+filename_out+str(img_nbr)+extension)
                print img_nbr

def rename_pressure_ramp(path_in,filename_in,path_out,filename_out,step,number_init,nbrOfDigits,extension=".bmp",descending=True):

    if descending:
        for filename in os.listdir(path_in):
            if filename.startswith(filename_in) and filename.endswith('.bmp'):
                formater = '{0:0'+nbrOfDigits+'d}'
                img_nbr =int(filename.split(".")[0].split("_")[-1])
                img_nbr = number_init-step*(img_nbr-1)
                img_nbr =  formater.format(img_nbr)
                shutil.copyfile(path_in+"\\"+filename, path_out+"\\"+filename_out+img_nbr+extension)
                print img_nbr
    else:
        for filename in os.listdir(path_in):
            if filename.startswith(filename_in):
                formater = '{0:0'+nbrOfDigits+'d}'
                img_nbr =int(filename.split(".")[0].split("_")[-1])
                img_nbr = number_init+step*(img_nbr-1)
                if (img_nbr%step ==0):
                    img_nbr =  formater.format(img_nbr)
                    shutil.copyfile(path_in+"\\"+filename, path_out+"\\"+filename_out+img_nbr+extension)
                print img_nbr

def convert_image_16bit(path_in,filename_in,type=np.uint16):
    i=0
    for filename in os.listdir(path_in):
            if filename.startswith(filename_in):
                img_in = cv2.imread(path_in+filename,1)
                print img_in.shape
                print img_in.dtype
                img_out = np.array(img_in, type)
                print img_out.dtype
                filename_out = 'converted_'+filename
                # cv2.imshow("test",img_out)
                # cv2.waitKey()
                cv2.imwrite(path_in+filename_out,img_out)
                print i
                i+=1

def convert_image_bitmap(path_in,extension):
    i=0
    for filename in os.listdir(path_in):
            if filename.endswith(extension):
                img_in = cv2.imread(path_in+filename,1)
                filename_out =filename.replace(extension,".bmp",1)
                cv2.imwrite(path_in+filename_out,img_in)

                print i
                i+=1

if __name__=="__main__":
    if PROCESS:
        path_in = "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\2_25\\Quasi_static\\Manip_1\\Buckling\\Isotropic\\"

        path_out = "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\5_25\\calibration\\"
        filename_in = "converted_pressure_"
        filename_out = "converted_pressure_"
        step = 1
        number_out = 0
        number_out = image_stack_processor(path_in,filename_in,path_out,filename_out,step,number_out)
        filename_in = "RedBall_300_0_registered"
        number_out = image_stack_processor(path_in,filename_in,path_out,filename_out,step,number_out)
    if RENAME:
        path_in = "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\2_25\\Dynamic\\Manip_4\\Backward\\Rolling\\Registered\\"
        path_out = "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\2_25\\Dynamic\\Manip_4\\Backward\\Rolling\\Registered\\"
        00001
        filename_in = ""
        filename_out = "p_"
        step = 1
        number_init = 200
        rename_image(path_in,filename_in,path_out,filename_out,step,number_init,de)
    if RENAMERAMP:
        path_in =  "D:\\Documents\\Footage\\videos\\Manip_Bruno\\Ajo_122\\Dynamic\\Manip_2\\Backward\\Rolling\\P_variable\\"
        path_out = "D:\\Documents\\Footage\\videos\\Manip_Bruno\\Ajo_122\\Dynamic\\Manip_2\\Backward\\Rolling\\P_variable\\"
        filename_in = "p_300_150_"
        filename_out = "p_"
        step = 1
        number_init = 300
        nbrOfDigits = '3'
        rename_pressure_ramp(path_in,filename_in,path_out,filename_out,step,number_init,nbrOfDigits=nbrOfDigits,descending=True)
    if CONVERT:
        path_in = "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\6.5_25\\Calibration\\"
        path_out = "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\6.5_25\\Calibration\\"
        filename_in = "p_"
        convert_image_16bit(path_in,filename_in)
    if BITMAP:
        path_in = "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\5_25\\Quasi_static\\Manip_1\\Buckling\\Isotropic\\"
        path_out = "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\5_25\\Quasi_static\\Manip_1\\Buckling\\Isotropic\\"
        extension = ".jpg"
        convert_image_bitmap(path_in,extension)

    # if number_out == 0:
    #     number_out = 700
    # for i in range(700,930,10):
    #     sub_dir_name = str(i)+"_"+str(i+10)
    #     path_in_sub = path_in +"\\"+ sub_dir_name
    #     filename_in = sub_dir_name+ "_1mbar_100fps_"
    #     number_out = image_stack_processor(path_in_sub,filename_in,path_out,filename_out,100,number_out,10)
    # number_out=930
    # number_out = image_stack_processor(path_in_sub,filename_in,path_out,filename_out,1,number_out)