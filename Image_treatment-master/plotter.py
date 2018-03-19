import numpy as np
from matplotlib import pyplot as plt
import os


# ##5
Yg_reference = [2.57520e-02,2.58761e-02,2.59152e-02,2.57457e-02,2.57457e-02]
ly_reference = [5.10853e-02,5.10960e-02,5.12694e-02,5.10834e-02,5.10834e-02]
ext_V_reference =[6.54484e-05,6.54422e-05,6.54462e-05,6.54477e-05,6.54480e-05]
timeStartbuckling = [1500,1000,163,1821,435]
timeProgressionRolling = [False,False,True,True,True]
IndexPressure = 340
timStartArraybuckling = np.array(timeStartbuckling)*0.0002
timeStopArraybuckling = timStartArraybuckling+0.1
timeStartdebuckling = [5850,5850,180,161,1665]
timStartArraydebuckling = np.array(timeStartdebuckling)*0.0002
timeStopArraydebuckling = timStartArraydebuckling+0.1
pressure_ref = 101300
#



# # #6.5
# Yg_reference = [2.63940e-02,2.75485e-02,2.76895e-02,2.72654e-02,2.74623e-02]
# ly_reference = [5.14205e-02,5.25599e-02,5.27431e-02,5.23628e-02,5.25493e-02]
# ext_V_reference =[6.54476e-05,6.54452e-05,6.54495e-05,6.54458e-05,6.54447e-05]
# timeStartbuckling = [1530,1650,1435,975,1718]
# timeProgressionRolling = [False,False,True,True,True]
# # IndexPressure = 340
# timStartArraybuckling = np.array(timeStartbuckling)*0.0002
# timeStopArraybuckling = timStartArraybuckling+0.1
# timeStartdebuckling = [2765,3985,4240,3925,2830]
# timStartArraydebuckling = np.array(timeStartdebuckling)*0.0002
# timeStopArraydebuckling = timStartArraydebuckling+0.1
# pressure_ref = 101300
#
#
# #2
# Yg_reference = [2.62422e-02,2.62123e-02,2.64830e-02,2.65178e-02,2.65126e-02]
# ly_reference = [5.24448e-02,5.22848e-02,5.27078e-02,5.27705e-02,5.28126e-02]
# ext_V_reference =[6.54160e-05,6.54171e-05,6.54485e-05,6.52726e-05,6.54427e-05]
# timeStartbuckling = [1530,1550,922,1000,30]
# timeProgressionRolling = [False,False,True,True,True]
# timStartArraybuckling = np.array(timeStartbuckling)*0.0002
# timeStopArraybuckling = timStartArraybuckling+0.1
# timeStartdebuckling = [4255,4280,4480,2330,1370]
# timStartArraydebuckling = np.array(timeStartdebuckling)*0.0002
# timeStopArraydebuckling = timStartArraydebuckling+0.1
# pressure_ref = 101300

def file_reader(name,type):
    f = open(name,"r")
    lines = f.readlines()
    print name
    if type=="results":
        timeList = []
        pressureList = []
        deltaPList = []
        lowPointList = []
        gravity_yList = []
        externalVList = []
        elements = lines[0].split('\t')
        image_name = False
        if elements[2]=="image_name":
            image_name =True
        if image_name:
            for e in lines[1:]:
                if len(e)<10:
                    pass
                else:
                    elements = e.split('\t')
                    center_of_gravity =float(elements[6])
                    volume = float(elements[7])
                    ylow = float(elements[4])
                    test01 = center_of_gravity>=0
                    #if not test01:
                        #print "center of gravity negative"
                    test02 = center_of_gravity< 0.1
                    #if not test02:
                        #print "center of gravity higher than thresh"
                    test03 = volume>1e-05
                    #if not test03:
                        #print "volume lower than thresh"
                    test04 = volume<10e-05
                    #if not test04:
                        #print "volume higher than thresh"
                    test05 = ylow>0
                    #if not test05:
                        #print "ylow negative"
                    test06 = ylow<0.1
                    #if not test06:
                        #print "ylow higher than thresh"
                    test = test01 and test02 and test03 and test04 and test05 and test06
                    if test:
                        timeList.append(float(elements[0]))
                        pressureList.append(float(elements[1]))
                        deltaPList.append(float(3))
                        lowPointList.append(float(elements[4]))
                        gravity_yList.append(float(elements[6]))
                        externalVList.append(float(elements[7]))
        else:
            for e in lines[1:]:
                if len(e)<10:
                    pass
                else:
                    elements = e.split('\t')
                    center_of_gravity =float(elements[5])
                    volume = float(elements[6])
                    ylow = float(elements[3])
                    test01 = center_of_gravity>=0
                    #if not test01:
                        #print "center of gravity negative"
                    test02 = center_of_gravity< 0.1
                    #if not test02:
                        #print "center of gravity higher than thresh"
                    test03 = volume>1e-05
                    #if not test03:
                        #print "volume lower than thresh"
                    test04 = volume<10e-05
                    #if not test04:
                        #print "volume higher than thresh"
                    test05 = ylow>0
                    #if not test05:
                        #print "ylow negative"
                    test06 = ylow<0.1
                    #if not test06:
                        #print "ylow higher than thresh"
                    test = test01 and test02 and test03 and test04 and test05 and test06
                    if test:
                        timeList.append(float(elements[0]))
                        pressureList.append(float(elements[1]))
                        deltaPList.append(float(2))
                        lowPointList.append(float(elements[3]))
                        gravity_yList.append(float(elements[5]))
                        externalVList.append(float(elements[6]))
        return np.array(timeList),np.array(pressureList),np.array(deltaPList),np.array(lowPointList),np.array(gravity_yList),np.array(externalVList)

def plot_multiple(datax,datay,numberOfexperiments,outputFilename,title,xlabel,ylabel,show_flag=False,isotropic= False):
    if not isotropic:
        color_codes=["mo","co","ro","bo","go"]
        fig = plt.figure(figsize=(25,17), dpi=100)
        fig.autolayout = True
        for i in xrange(len(datax)):
            labelname = "Manip"+str(i+1)
            plt.plot(datax[i],datay[i],color_codes[i],label=labelname,markersize = 10)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        lgd = plt.legend(bbox_to_anchor=(1, 1), loc=2, borderpad=1.,fancybox=True,markerscale=1,fontsize=18)
        plt.grid()
        plt.savefig(outputFilename, bbox_inches='tight')
        if show_flag:
            plt.show()
        plt.close()
    else:
        color_codes = ["m","c","r","b","g"]
        sign_codes=["o","*","^","s","d"]
        experiment_parts = ["iso_b","iso_db","Equi_b","Equi_db","rolling"]
        fig = plt.figure(figsize=(25, 17), dpi=200)
        fig.autolayout = True
        nbr_of_parts = numberOfexperiments[0]
        for i in xrange(len(datax)):
            if i>=2*numberOfexperiments[0]:
                nbr_of_parts = numberOfexperiments[1]
                manip_nbr = 3+(i-(2*numberOfexperiments[0]))/nbr_of_parts
                part_nbr = (i-(2*numberOfexperiments[0]))%nbr_of_parts
            else:
                manip_nbr = 1+(i/nbr_of_parts)
                part_nbr = i%nbr_of_parts
            labelname = str(manip_nbr)+"_"+experiment_parts[part_nbr]
            try:
                plt.plot(datax[i],datay[i],color_codes[manip_nbr-1]+sign_codes[part_nbr],label=labelname,markersize = 10)

            except:
                print 'here'
                print i
                print nbr_of_parts
                print manip_nbr
                print part_nbr
                print len(datax)
                print len(datay)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        lgd = plt.legend(bbox_to_anchor=(1, 1), loc=2, borderpad=1.,fancybox=True,markerscale=1,fontsize=18)
        plt.grid()
        plt.savefig(outputFilename, bbox_inches='tight')
        if show_flag:
            plt.show()
        plt.close()

def plot_unique(datax,datay,outputFilename,graphlabel,graphtype,title,xlabel,ylabel,show_flag=False):
    # try:
    #     datax_min = np.amin(datax)
    # except:
    #     print datax
    #     print datay
    #     print outputFilename+" problem"
    # datax_max = np.amax(datax)
    # step_x = (datax_max-datax_min)/float(datax.shape[0])
    # datay_min = np.amin(datay)
    #
    # datay_max = np.amax(datay)


    # step_y = (datay_max-datay_min)/float(datay.shape[0])
    # print outputFilename
    # print datax_min-step_x,datax_max+step_x
    # print datay_min-step_y,datay_max+step_y
    #
    fig = plt.figure(figsize=(25, 17), dpi=200)
    # plt.xlim(datax_min-step_x,datax_max+step_x)
    # plt.ylim(datay_min-step_y,datay_max+step_y)
    plt.plot(datax,datay,graphtype,label=graphlabel,markersize = 10)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    lgd = plt.legend(bbox_to_anchor=(1, 1), loc=2, borderpad=1.,fancybox=True,markerscale=1,fontsize=18)
    plt.grid()
    plt.savefig(outputFilename, bbox_inches='tight')
    if show_flag:
        plt.show()
    plt.close()

def plot_double(datax,datay1,datay2,graphlabel1,graphlabel2,outputFilename,title,xlabel,ylabel,show_flag=False):
    fig = plt.figure(figsize=(17, 17), dpi=200)
    fig.autolayout = True
    plt.plot(datax,datay1,'r--',label=graphlabel1,markersize = 10)
    plt.plot(datax,datay2,'b--',label=graphlabel2,markersize = 10)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    lgd = plt.legend(bbox_to_anchor=(1, 1), loc=2, borderpad=1.,fancybox=True,markerscale=1,fontsize=18)
    plt.grid()
    plt.savefig(outputFilename, bbox_inches='tight')
    if show_flag:
        plt.show()
    plt.close()

def rolling(source,numberOfexperiments,ballname):
    filesList = os.listdir(source)
    outputFilesList = []
    for e in filesList:
        if e.endswith('.txt'):
            outputFilesList.append(e)


    timeArrayList = []
    pressureArrayList = []
    lowPointArrayList = []
    gravity_yArrayList = []
    externalVArrayList = []
    lowPointArraySameRefList = []
    gravity_yArraySameRefList = []
    print "reading"
    for i in xrange(numberOfexperiments):
        type="results"
        index = 0
        try:
            res = file_reader(source+outputFilesList[i],type)

            test01 = np.argwhere(ly_reference[i]-res[3][index:]<=0)
            test02 = np.argwhere(Yg_reference[i]-res[4][index:]<=0)
            test03 = np.argwhere(1-(res[5][index:]/ext_V_reference[i])>0)
            test04 = np.argwhere(1-(res[5][index:]/ext_V_reference[i])<0.1)
            test1 = np.intersect1d(test01,test02)
            test2 = np.intersect1d(test03,test04)
            test = np.intersect1d(test1,test2)
            timeArrayList.append(res[0][test])
            pressureArrayList.append((res[1][test]-pressure_ref)/100)
            lowPointArrayList.append(ly_reference[i]-res[3][test])
            gravity_yArrayList.append(Yg_reference[i]-res[4][test])
            externalVArrayList.append(1-res[5][test]/ext_V_reference[i])
            lowPointArraySameRefList.append(0.1-res[3][test])
            gravity_yArraySameRefList.append(0.1-res[4][test])
        except:
            pass

    for i in xrange(len(timeArrayList)):
        graphtitle = ' Manip_'+str(i+1)
        outputname_complete =source+ graphtitle+"Yg(P)vsYlow(P)_sameReferential"+format
        title=ballname+graphtitle+'gravity_center(t)'
        ylabel='deltaX (m)'
        ylabel1='deltaX_Yg (m)'
        ylabel2='deltaX_Ybottom (m)'
        xlabel='Pressure (mbar)'
        plot_double(pressureArrayList[i],gravity_yArraySameRefList[i],lowPointArraySameRefList[i],ylabel1,ylabel2,outputname_complete,title,xlabel,ylabel)


    ylabel='deltaX (m)'
    xlabel='Applied Pressure (mbar)'
    title=ballname+'Y_g(P) for different velocities'
    savefigname =source+"Y_g(P) for different velocities"+format
    plot_multiple(pressureArrayList,gravity_yArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)


    ylabel='deltaX_ball_bottom (m)'
    title=ballname+'Y_low(P) for different velocities'
    savefigname= source+"Y_low(P) for different velocities"+format
    plot_multiple(pressureArrayList,lowPointArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)

    ylabel='dV/V'
    title =ballname+'ExternalVolume for different velocities'
    savefigname = source+"ExtV for different velocities"+format
    plot_multiple(pressureArrayList,externalVArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)

    xlabel='dV/V'
    ylabel= 'deltaX (m)'
    title =ballname+'dX(V)'
    savefigname = source+"dx(V) for different velocities"+format
    plot_multiple(externalVArrayList,gravity_yArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)


    # for i in xrange(len(timeArrayList)):
    #     pressureArrayList.remove((res[1]/100)-pressure_ref)
    #     lowPointArrayList.append(ly_reference[i]-res[3])
    #     gravity_yArrayList.append(Yg_reference[i]-res[4])
    #     externalVArrayList.append(1-res[5]/ext_V_reference[i])
    # xlabel = 'Time (s)'
    # ylabel='deltaX (m)'
    # ='Y_g(t) for constantPressure'
    # savefigname = source+"Y_g(t) for constantPressure"
    # plot_multiple(timeArrayList,externalVArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)

def dynamic(source,numberOfexperiments,startPerExperiment,deltaT,ballname):

    filesList = os.listdir(source)
    outputFilesList = []
    for e in filesList:
        if e.endswith('.txt'):
            outputFilesList.append(e)
    timeArrayList = []
    pressureArrayList = []
    deltaPArrayList = []
    lowPointArrayList = []
    gravity_yArrayList = []
    externalVArrayList = []
    lowPointArraySameRefList = []
    gravity_yArraySameRefList = []
    print "reading"
    for i in xrange(numberOfexperiments):
        type="results"
        index = 0


        res = file_reader(source+outputFilesList[i],type)

        test01 = np.argwhere(ly_reference[i]-res[3][index:]<=0)
        test02 = np.argwhere(Yg_reference[i]-res[4][index:]<=0)
        test03 = np.argwhere(1-(res[5][index:]/ext_V_reference[i])>0)
        test04 = np.argwhere(1-(res[5][index:]/ext_V_reference[i])<0.5)
        test1 = np.intersect1d(test01,test02)
        test2 = np.intersect1d(test03,test04)
        test = np.intersect1d(test1,test2)

        timeArrayList.append(res[0][test])
        pressureArrayList.append((res[1][test]-pressure_ref)/100)
        deltaPArrayList.append(res[2][test])
        lowPointArrayList.append(ly_reference[i]-res[3][test])
        gravity_yArrayList.append(Yg_reference[i]-res[4][test])
        externalVArrayList.append(1-res[5][test]/ext_V_reference[i])
        lowPointArraySameRefList.append(0.1-res[3][test])
        gravity_yArraySameRefList.append(0.1-res[4][test])


    color_codes=["mo","co","ro","bo","go"]
    timeStartArrayList = []
    timeZoomArrayList = []
    gravity_yStartArrayList = []
    gravity_yZoomArrayList = []
    lowPointStartArrayList = []
    lowPointZoomArrayList  = []
    R_ArrayList = []
    R_startArrayList = []
    R_ZoomArrayList = []
    externalVStartArrayList = []
    externalVZoomArrayList = []
    lowPointArraySameRefListStart = []
    gravity_yArraySameRefListStart = []
    lowPointArraySameRefListZoom = []
    gravity_yArraySameRefListZoom = []

    for i in xrange(len(timeArrayList)):
        istart = np.amin(np.argwhere(timeArrayList[i]>startPerExperiment[i]))
        istop = np.amin(np.argwhere(timeArrayList[i]>deltaT[i]))
        timeStartArrayList.append(timeArrayList[i][istart:]-timeArrayList[i][istart])
        timeZoomArrayList.append(timeArrayList[i][istart:istop]-timeArrayList[i][istart])

        gravity_yStartArrayList.append(gravity_yArrayList[i][istart:])
        gravity_yZoomArrayList.append(gravity_yArrayList[i][istart:istop])

        lowPointStartArrayList.append(lowPointArrayList[i][istart:])
        lowPointZoomArrayList.append(lowPointArrayList[i][istart:istop])

        R = gravity_yArrayList[i]-lowPointArrayList[i]
        R_ArrayList.append(R)
        R_startArrayList.append(R_ArrayList[i][istart:])
        R_ZoomArrayList.append(R_ArrayList[i][istart:istop])
        externalVStartArrayList.append(externalVArrayList[i][istart:])
        externalVZoomArrayList.append(externalVArrayList[i][istart:istop])

        lowPointArraySameRefListStart.append(lowPointArraySameRefList[i][istart:])
        gravity_yArraySameRefListStart.append(gravity_yArraySameRefList[i][istart:])
        lowPointArraySameRefListZoom.append(lowPointArraySameRefList[i][istart:istop])
        gravity_yArraySameRefListZoom.append(gravity_yArraySameRefList[i][istart:istop])

    for i in xrange(len(timeArrayList)):
        graphtitle = ' Manip_'+str(i+1)
        outputname_complete =source+ graphtitle+"gravity_center(t)_complete"+format
        title = ballname+graphtitle+' gravity_center(t)'
        ylabel='deltaX (m)'
        xlabel='Time (s)'
        plot_unique(timeArrayList[i],gravity_yArrayList[i],outputname_complete,graphtitle,color_codes[i],title,xlabel,ylabel)
        outputname_start =source+ graphtitle+"gravity_center(t)_start_correction"+format
        plot_unique(timeStartArrayList[i],gravity_yStartArrayList[i],outputname_start,graphtitle,color_codes[i],title,xlabel,ylabel)
        outputname_zoom = source+ graphtitle+"gravity_center(t)_zoom"+format
        plot_unique(timeZoomArrayList[i],gravity_yZoomArrayList[i],outputname_zoom,graphtitle,color_codes[i],title,xlabel,ylabel)

    for i in xrange(len(timeArrayList)):
        graphtitle = ' Manip_'+str(i+1)
        outputname_complete =source+ graphtitle+"Yg(t)vsYlow(t)_sameReferential"+format
        title = ballname+graphtitle+' gravity_center(t)'
        ylabel='deltaX (m)'
        ylabel1='deltaX_Yg (m)'
        ylabel2='deltaX_Ybottom (m)'
        xlabel='Time (s)'
        plot_double(timeArrayList[i],gravity_yArraySameRefList[i],lowPointArraySameRefList[i],ylabel1,ylabel2,outputname_complete,title,xlabel,ylabel)
        outputname_start =source+ graphtitle+"Yg(t)vsYlow(t)_sameReferential_start_correction"+format
        plot_double(timeStartArrayList[i],gravity_yArraySameRefListStart[i],lowPointArraySameRefListStart[i],ylabel1,ylabel2,outputname_start,title,xlabel,ylabel)
        outputname_zoom = source+ graphtitle+"Yg(t)vsYlow(t)_sameReferential_zoom"+format
        plot_double(timeZoomArrayList[i],gravity_yArraySameRefListZoom[i],lowPointArraySameRefListZoom[i],ylabel1,ylabel2,outputname_zoom,title,xlabel,ylabel)

    ylabel='deltaX (m)'
    xlabel='Time(s)'
    title=ballname+' Y_g(t)'

    savefigname =source+"Y_g(t)_complete"+format
    plot_multiple(timeArrayList,gravity_yArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)
    savefigname =source+"Y_g(t)_start_correction"+format
    plot_multiple(timeStartArrayList,gravity_yStartArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)
    savefigname =source+"Y_g(t)_zoom"+format
    plot_multiple(timeZoomArrayList,gravity_yZoomArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)

    for i in xrange(len(timeArrayList)):
        graphtitle = ' Manip_'+str(i+1)
        outputname_complete =source+ graphtitle+"lowPoint(t)_complete"+format
        title = ballname+graphtitle+' lowPoint(t)'
        ylabel='deltaX_ball_bottom (m)'
        xlabel='Time (s)'

        plot_unique(timeArrayList[i],lowPointArrayList[i],outputname_complete,graphtitle,color_codes[i],title,xlabel,ylabel)
        outputname_start =source+ graphtitle+"lowPoint(t)_start_correction"+format
        plot_unique(timeStartArrayList[i],lowPointStartArrayList[i],outputname_start,graphtitle,color_codes[i],title,xlabel,ylabel)
        outputname_zoom = source+ graphtitle+"lowPoint(t)_zoom"+format
        plot_unique(timeZoomArrayList[i],gravity_yZoomArrayList[i],outputname_zoom,graphtitle,color_codes[i],title,xlabel,ylabel)

    ylabel='deltaX_ball_bottom (m)'
    xlabel='Time(s)'
    title=ballname+' Y_low(t)'
    savefigname= source+"Y_low(t)_complete"+format
    plot_multiple(timeArrayList,lowPointArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)
    savefigname= source+"Y_low(t)_start_correction"+format
    plot_multiple(timeStartArrayList,lowPointStartArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)
    savefigname =source+"Y_low(t)_zoom"+format
    plot_multiple(timeZoomArrayList,lowPointZoomArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)

    for i in xrange(len(timeArrayList)):
        graphtitle = ' Manip_'+str(i+1)
        outputname_complete =source+ graphtitle+"R(t)_complete"+format
        title=ballname+graphtitle+' R(t)'
        ylabel='deltaR (m)'
        xlabel='Time (s)'

        plot_unique(timeArrayList[i],R_ArrayList[i],outputname_complete,graphtitle,color_codes[i],title,xlabel,ylabel)
        outputname_start =source+ graphtitle+"R(t)_start_correction"+format
        plot_unique(timeStartArrayList[i],R_startArrayList[i],outputname_start,graphtitle,color_codes[i],title,xlabel,ylabel)
        outputname_zoom = source+ graphtitle+"R(t)_zoom"+format
        plot_unique(timeZoomArrayList[i],R_ZoomArrayList[i],outputname_zoom,graphtitle,color_codes[i],title,xlabel,ylabel)

    ylabel='R (m)'
    xlabel='Time(s)'
    title=ballname+' R(t)'
    savefigname= source+"R(t)_complete"+format
    plot_multiple(timeArrayList,R_ArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)
    savefigname= source+"R(t)_start_correction"+format
    plot_multiple(timeStartArrayList,R_startArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)
    savefigname =source+"R(t)_zoom"+format
    plot_multiple(timeZoomArrayList,R_ZoomArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)

    for i in xrange(len(timeArrayList)):
        graphtitle = ' Manip_'+str(i+1)
        outputname_complete =source+ graphtitle+"external_V(t)_complete"+format
        title=ballname+graphtitle+' V(t)'
        ylabel='dV/V'
        xlabel='Time (s)'

        plot_unique(timeArrayList[i],externalVArrayList[i],outputname_complete,graphtitle,color_codes[i],title,xlabel,ylabel)
        outputname_start =source+ graphtitle+"external_V(t)_start_correction"+format
        plot_unique(timeStartArrayList[i],externalVStartArrayList[i],outputname_start,graphtitle,color_codes[i],title,xlabel,ylabel)
        outputname_zoom = source+ graphtitle+"external_V(t)_zoom"+format
        plot_unique(timeZoomArrayList[i],externalVZoomArrayList[i],outputname_zoom,graphtitle,color_codes[i],title,xlabel,ylabel)

    ylabel='dV/V (m^3)'
    xlabel='Time (s)'
    title =ballname+'ExternalVolume(t)'
    savefigname = source+"ExtV(t)_complete"+format
    plot_multiple(timeArrayList,externalVArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)
    savefigname = source+"ExtV(t)_start_correction"+format
    plot_multiple(timeStartArrayList,externalVStartArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)
    savefigname = source+"ExtV(t)_zoom"+format
    plot_multiple(timeZoomArrayList,externalVZoomArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,False)

def isotropic(source,numberOfexperiments,ballname):
    filesList = os.listdir(source)
    outputFilesList = []
    for e in filesList:
        if e.endswith('.txt'):
            outputFilesList.append(e)
    timeArrayList = []
    pressureArrayList = []
    deltaPArrayList = []
    lowPointArrayList = []
    gravity_yArrayList = []
    externalVArrayList = []
    print "isotropic reading"
    for i in xrange(len(outputFilesList)):
        type="results"
        try:
            if i<2*numberOfexperiments[0]:
                k = i/numberOfexperiments[0]
            else:
                k = (i-(2*numberOfexperiments[0]))/numberOfexperiments[1]+2
            res = file_reader(source+outputFilesList[i],type)
            index = 0
            test01 = np.argwhere(ly_reference[k]-res[3][index:]<=0)
            test02 = np.argwhere(Yg_reference[k]-res[4][index:]<=0)
            test03 = np.argwhere(1-(res[5][index:]/ext_V_reference[k])>0)
            test04 = np.argwhere(1-(res[5][index:]/ext_V_reference[k])<0.1)
            test1 = np.intersect1d(test01,test02)
            test2 = np.intersect1d(test03,test04)
            test = np.intersect1d(test1,test2)
            timeArrayList.append(res[0][test])
            pressureArrayList.append((res[1][test]-pressure_ref)/100)
            deltaPArrayList.append(res[2][test])
            print outputFilesList[i]
            print ly_reference[k]
            lowPointArrayList.append(ly_reference[k]-res[3][test])
            gravity_yArrayList.append(Yg_reference[k]-res[4][test])
            externalVArrayList.append(res[5][test])
        except:
            pass
    print len(timeArrayList)
    print len(pressureArrayList)
    print len(deltaPArrayList)
    print len(lowPointArrayList)
    print len(gravity_yArrayList)
    print len(externalVArrayList)
    ylabel='deltaX (m)'
    xlabel='Applied Pressure (mbar)'
    title=ballname+' Y_g(P) comparison'
    savefigname =source+"Y_g(P) comparison"+format
    plot_multiple(pressureArrayList,gravity_yArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,show_flag=False,isotropic=True)


    ylabel='deltaX_ball_bottom (m)'
    title=ballname+' Y_low(P) comparison'
    savefigname= source+"Y_low(P) comparison"+format
    plot_multiple(pressureArrayList,lowPointArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,show_flag=False,isotropic=True)

    ylabel='V (m^3)'
    title =ballname+' ExternalVolume comparison'
    savefigname = source+"ExtV comparison"+format
    plot_multiple(pressureArrayList,externalVArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,show_flag=False,isotropic=True)

    xlabel='V (m^3)'
    ylabel= 'deltaX (m)'
    title =ballname+' dX(V)'
    savefigname = source+"dx(V) comparison"+format
    plot_multiple(externalVArrayList,gravity_yArrayList,numberOfexperiments,savefigname,title,xlabel,ylabel,show_flag=False,isotropic=True)

format = ".png"
nbr_experiments = 5
ball_name = "5_25"
source_rolling= "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\"+ball_name+"\\Output_general\\Rolling\\"
rolling(source_rolling,nbr_experiments,ball_name)
source_buckling= "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\"+ball_name+"\\Output_general\\Buckling\\"
dynamic(source_buckling,nbr_experiments,timStartArraybuckling,timeStopArraybuckling,ball_name)
source_debuckling= "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\"+ball_name+"\\Output_general\\Debuckling\\"
dynamic(source_debuckling,nbr_experiments,timStartArraydebuckling,timeStopArraydebuckling,ball_name)
source_b_isotropic= "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\"+ball_name+"\\Output_general\\Isotropic_B\\"
nbr_experiments = (5,4)
isotropic(source_b_isotropic,nbr_experiments,ball_name)
# source_db_isotropic= "D:\\Documents\\Footage\\videos\\Manip_finales\\Glycerol_100\\Ressort\\2_25\\Output_general\\Isotropic_Db\\"
# isotropic(source_db_isotropic,nbr_experiments)