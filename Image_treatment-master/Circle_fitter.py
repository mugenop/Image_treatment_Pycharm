__author__ = 'Adel'
from PIL import Image
from numpy import *
from matplotlib import pyplot as p
from Pressure_file_structurer import *

MAX_IMAGE_NUMBER = 10
MIN_IMAGE_NUMBER = 0
THRESHOLD_DIAMETER_FILTER = 268
THRESHOLD_PIXEL = 10
THRESHOLD_MIN_X = 20
THRESHOLD_MAX_X = 710
THRESHOLD_MIN_Y = 23
THRESHOLD_MAX_Y = 960
THRESHOLD_VENTOUSE_FILTER = 90
THRESHOLD_VENTOUSE_FILTER_2 = 133

FPS = 1
# ============================================================================
from scipy import optimize
import functools

def countcalls(fn):
    "decorator function count function calls "


    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.ncalls +=1
        return fn(*args)

    wrapped.ncalls = 0
    return wrapped
method_2 = "leastsq"

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return sqrt((x-xc)**2 + (y-yc)**2)

@countcalls
def algebraic_distance(c):
    """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()
@countcalls
def Df_2b(c):
    """ Jacobian of f_2b
    The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
    xc, yc = c
    df2b_dc = empty((len(c), x.size))

    Ri = calc_R(xc, yc)
    df2b_dc[ 0] = (xc - x)/Ri                   # dR/dxc
    df2b_dc[ 1] = (yc - y)/Ri                   # dR/dyc
    df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, newaxis]

    return df2b_dc

# ============================================================================


def contour_finder(pix2D):
    maxDistances = []
    for v in range(THRESHOLD_MIN_Y, THRESHOLD_MAX_Y):
        blacks=[]
        for u in range(THRESHOLD_MIN_X, THRESHOLD_MAX_X):
            if pix2D[u, v] != (255, 255, 255):
                blacks.append(u)
        if len(blacks) != 0:
            d = blacks[-1]-blacks[0]
            centerX=blacks[0]+d/2
            maxDistances.append((d, centerX, v, blacks[-1], blacks[0]))
    return maxDistances


def maxD_finder(maxDistances):
    max = 0
    finalD = (0, 0, 0)
    elIndex = 0
    for el in maxDistances:
        if el[0] >= max:
            max = el[0]
            finalD = el
            elIndex = maxDistances.index(el)
    return finalD, elIndex


def pixel_delute(pix2D,x_c,y_c,edge,color):
    for v in range(y_c-edge, y_c+edge):
        for u in range(x_c-edge, x_c+edge):
            pix2D[u, v] = color


def minD_finder(maxDistances):
    finalD, elIndex = maxD_finder(maxDistances)
    min = 100000
    for el in maxDistances:
        if maxDistances.index(el)>elIndex:
            if el[0] < min:
                min = el[0]
                finalD = el
                elIndex = maxDistances.index(el)
    return finalD, elIndex


def shorter(maxDistances):
    counter = 0
    finalD, elIndex = maxD_finder(maxDistances)
    # print elIndex,len(maxDistances)
    for el in maxDistances:
        gamma = maxDistances.index(el)
        if maxDistances.index(el) > elIndex:
            DSK = el[0]
            if el[0] < THRESHOLD_DIAMETER_FILTER:
                counter = maxDistances.index(el)
                break
    endmaxDiameters = maxDistances[counter:len(maxDistances)-1]
    del maxDistances[counter:len(maxDistances)-1]
    return maxDistances,endmaxDiameters


def x_y_mapper (maxDistances):
    x = []
    y = []

    for el in maxDistances:
        x.append(el[4])
        x.append(el[3])
        y.append(el[2])
        y.append(el[2])
    x= array(x)
    y= array (y)
    x_m = mean(x)
    y_m = mean(y)
    Xinterger= int(x_m)
    Yinteger= int(y_m)

    return x, y, x_m, y_m, Xinterger, Yinteger


def maxDistances_color(maxDistances,pix2D,color):
    for el in maxDistances:
            pix2D[el[3],el[2]] = color
            pix2D[el[4],el[2]] = color


def ball_colorer(x_c, y_c, R, pix2D):
    for v in range(int(y_c-R), int(y_c+R)):
        for u in range(int(x_c-R), int(x_c+R)):
            if( (u-x_c)**2+(v-y_c)**2-(R)**2<=0):
                pix2D[u,v] = (0,204,0)


def threshold_color(pix2D, color):
    for v in range(THRESHOLD_MIN_Y, THRESHOLD_MAX_Y):
        pix2D[THRESHOLD_MIN_X, v] = color
        pix2D[THRESHOLD_MAX_X, v] = color
    for u in range(THRESHOLD_MIN_X, THRESHOLD_MAX_X):
        pix2D[u, THRESHOLD_MIN_Y] = color
        pix2D[u, THRESHOLD_MAX_Y] = color


def ventouse_finder(list,limit):
    begin = list[0]
    end = list[-1]
    b = len (list)
    finalD, elIndex = maxD_finder(list)
    # print elIndex,len(maxDistances)
    counter = 0
    for el in list:
        gamma = list.index(el)
        DSK = el[0]
        if el[0] < limit:
            counter = list.index(el)
            break
    endlist = list[counter:len(list)-1]
    a = len(endlist)
    del list[counter:len(list)-1]
    return list,endlist


def ventouse_bottom_finder(list):
    a = len(list)
    ventouse_list, endlist = ventouse_finder(list,THRESHOLD_VENTOUSE_FILTER)
    b= len(ventouse_list)
    c= len(endlist)
    cornerlist, edgeslist = ventouse_finder(ventouse_list,THRESHOLD_VENTOUSE_FILTER_2)
    d = len(cornerlist)
    e = len(edgeslist)
    maxi = 0
    y_position = 0
    for v in range(edgeslist[0][2], edgeslist[-1][2]):
        counter = 0
        for u in range(min(edgeslist[0][3],edgeslist[-1][3]), max(edgeslist[0][4],edgeslist[-1][4])):
            if pix2D[u, v] != (255, 255, 255):
                counter+=1
        if maxi<=counter:
            maxi = counter
            y_position = v

    return (min(edgeslist[0][3],edgeslist[-1][3])+ max(edgeslist[0][4],edgeslist[-1][4]))/2, y_position


def line_constructor(list):

    x_line, y_line= ventouse_bottom_finder(list)

    return x_line, y_line


def numpy_array_constructor(list):
    x_l = []
    y_l = []
    for el in list:
        x_l.append(el[3])
        y_l.append(el[2])
    return array(x_l),array(y_l)

def line_colorer(x_i, a, b, ranger, pix2D, color,dt=False):
    x_list = []
    if dt:

        for l in range(0,ranger*100):
            x_list.append(x_i+l/100)
            x_list.append(x_i-l/100)
    else:
        for l in range(0,ranger):
            x_list.append(x_i+l)
            x_list.append(x_i-l)
    x_array = array(x_list)
    y_array = map(lambda x: x*a + b, x_array)
    for d in range(0, len(x_array)):
        pixel_delute(pix2D, int(x_array[d]), int(y_array[d]),2, color)
Y_line = []
Diameters = []
Residus = []
for i in range(MIN_IMAGE_NUMBER, MAX_IMAGE_NUMBER):
    print i
    index = ''
    if i < 10:
        index = '000'+str(i)
    elif i < 100:
        index = '00'+str(i)
    elif i < 1000:
        index = '0'+str(i)
    else:
        index = str(i)
    ball = Image.open('outline/Pre_buckling'+index+'.bmp')
    ball = ball.convert("RGB")
    pix2D = ball.load()
    maxDistances = contour_finder(pix2D)
    # maxDistances_color(maxDistances,pix2D,(204,0,0))
    qqsdqsd = len(maxDistances)
    maxDistances, newList = shorter(maxDistances)
    # maxDistances_color(newList,pix2D,(0,204,0))


    pix_x, pix_y = line_constructor(newList)
    # line_colorer(pix_x,0,pix_y,50,pix2D,(125,75,150))
    # pixel_delute(pix2D, int(pix_x), int(pix_y), 2, (100, 100, 100))
    x, y, x_m, y_m, Xinteger, Yinteger = x_y_mapper(maxDistances)
    center_estimate = x_m, y_m
    center_2b, ier = optimize.leastsq(algebraic_distance, center_estimate, Dfun=Df_2b, col_deriv=True)

    # ===================================================================
    xc_2b, yc_2b = center_2b
    Ri_2b = calc_R(xc_2b, yc_2b)
    R_2b = Ri_2b.mean()
    finalD = (R_2b*2,xc_2b, yc_2b)
    residu_2b = sum((Ri_2b - R_2b)**2)
    residu2_2b = sum((Ri_2b**2-R_2b**2)**2)
    ncalls_2b = algebraic_distance.ncalls
    # ball_colorer(finalD[1], finalD[2], finalD[0]/2,pix2D)
    ball.save('output/filledBall'+index+'.bmp')
    Y_line.append(pix_y)
    Diameters.append(finalD)
    Residus.append(residu2_2b)
Ds = []
Xc= []
Yc = []
for el in Diameters:
    Ds.append(el[0])
    Xc.append(el[1])
    Yc.append(el[2])
lines, t, p = file_parser("0_700_out.txt")
if len(t)>len(Ds):
    del t[-(len(t)-len(Ds)):len(t)]
    del p[-(len(p)-len(Ds)):len(p)]
if len(t)<len(Ds):
    last_t= t[-1]
    last_p = p[-1]
    for i in range(len(t),len(Ds)):
        t.append(last_t+(i-len(t))*FPS)
    for i in range (len(p),len(Ds)):
        p.append(last_p)
p = array(p)
t = array(t)
pixTomm = 9.9/132.0
D = pixTomm*array(Ds)
R = array(Residus)
pixTomm = 9.9/132.0
X_center = pixTomm*(array(Xc)-Xc[0])
yc0 = Yc[0]
Y_center = pixTomm*(array(Yc)-yc0)
Y_lines = array(Y_line)
L0 = Y_lines[0]
spring_displacement = (Y_lines-L0)*pixTomm

# Dat_Xc_t = column_stack((t, X_center))
# Dat_Yc_t = column_stack((t, Y_center))
# Dat_D_t = column_stack((t, D))
# Dat_D_p = column_stack((p, D))
# Dat_Yc_p = column_stack((p, Y_center))
# Dat_spring_displacement_t = column_stack((t, spring_displacement))
# Dat_spring_displacement_p = column_stack((p, spring_displacement))

savetxt('output/data/D(t).txt', Dat_D_t, fmt='%1.3f\t%1.3f', header="time\tDiameter")
savetxt('output/data/delatYcenter(t).txt', Dat_Yc_t,fmt='%1.3f\t%1.3f', header="time\tdelta Y center")
savetxt('output/data/delatXcenter(t).txt', Dat_Xc_t,fmt='%1.3f\t%1.3f', header="time\tdelta X center")
savetxt('output/data/delatXcenter(t).txt', Dat_spring_displacement_t, fmt='%1.3f\t%1.3f', header="time\tdelta spring displacement")

savetxt('output/data/D(p).txt', Dat_D_p, fmt='%1.3f\t%1.3f',header="pressure\tDiameter" )
savetxt('output/data/delatYcenter(p).txt', Dat_Yc_p, fmt='%1.3f\t%1.3f', header="pressure\tdelta Y center")
savetxt('output/data/delatXcenter(p).txt', Dat_spring_displacement_p, fmt='%1.3f\t%1.3f', header="pressure\tdelta spring displacement")
p.plot(t,R)
p.show()