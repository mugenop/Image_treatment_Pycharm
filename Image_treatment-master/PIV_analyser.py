from shape_PIV import ShapePIV
import os
from manip_analyser import config_reader,array_constructor
import sys
def movie_mask(config_path,beg=0,show_flag=False,save_flag=False):
    config_parameters = config_reader(config_path)
    concavity_array = array_constructor(config_parameters['concavity'],boolean=True)
    input_dir = config_parameters['image_input']
    output_dir = config_parameters['image_output']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    images_list = os.listdir(input_dir)
    i = beg
    x0 = None
    while i<len(images_list):

        # try:
        print 'treating image '+ str(i)+' over '+str(len(images_list))
        concavity_flag = concavity_array[i]
        image_in_name = input_dir+images_list[i]
        image_out_name = output_dir+images_list[i]
        temp_shape = ShapePIV(image_in_name,image_out_name,config_parameters,concavity=concavity_flag)
        # temp_shape.show("output",temp_shape.in_img)
        temp_shape.shape_analyse(x0=x0,cartesian=True)
        if save_flag:
            temp_shape.save(image_out_name,temp_shape.in_img,)
        if show_flag:
            temp_shape.show("output",temp_shape.in_img)
        i+=1
        # except:
        #     print "Unexpected error:", sys.exc_info()[0]
        #     i+=1
        #     pass

if __name__ == "__main__":
    config_file = "PIV/PIV_Buckling_zoom_10k"
    beg = 3400
    movie_mask(config_file,beg= beg,save_flag=True,show_flag=True)