__author__ = 'Adel'
DT = 0.1


def time_offset_finder(line):
    return float(line.split("\t")[0])


def pressure_creator(dtfps, pressurelist, timelist):
    numberofaddedvalues = DT/dtfps
    newtimelist = []
    newpressurelist = []
    for k in range(0, len(pressurelist)):
        for i in range(0,int(numberofaddedvalues)):
            t = timelist[k]+i*dtfps
            newtimelist.append(timelist[k]+i*dtfps)
            p = pressurelist[k]
            newpressurelist.append(pressurelist[k])
    return newpressurelist, newtimelist


def pressure_remover(dtfps, pressurelist, timelist):
    numberofremovedvalues = int(dtfps/DT)
    newtimelist = []
    newpressurelist = []
    for i in range(0, len(pressurelist), numberofremovedvalues):
        newtimelist.append(timelist[i])
        newpressurelist.append(pressurelist[i])
    return newpressurelist, newtimelist


def file_parser(path_in):
    print "treating file: "+path_in
    fobject_in = open(path_in, "r")
    lines = fobject_in.readlines()

    times = []
    pressures = []
    timeoffset = time_offset_finder(lines[1])
    for line in lines:
        if lines.index(line) > 0:

            splited = line.split('\t')
            if len(splited[0])>0:
                t = float(splited[0])-timeoffset
                times.append(t)
                p = float(splited[1][0:-1])
                pressures.append(p)
    fobject_in.close()
    return lines, times, pressures


def file_processor(path_in, path_out, fps):
    pressures = []
    times = []
    lines, times, pressures = file_parser(path_in)
    fobject_out = open(path_out, "w")
    dtfps = 1/fps

    newtimelist = []
    newpressurelist = []
    if DT > dtfps:

        newpressurelist, newtimelist = pressure_creator(dtfps, pressures, times)
    else:
        newpressurelist, newtimelist = pressure_remover(dtfps, pressures, times)

    fobject_out.writelines(lines[0])
    out_lines = []
    for i in range(0, len(newpressurelist)):
        out_lines.append('{0}\t{1}\n'.format(newtimelist[i], newpressurelist[i]))
    fobject_out.writelines(out_lines)
    fobject_out.close()
#
# path_ins = ["0_700.txt", "650_720.txt", "720_350.txt", "350_300.txt"]
# path_outs = ["0_700_out.txt", "650_720_out.txt", "720_350_out.txt", "350_300_out.txt"]
# fps = [1.0, 100.0, 1, 100.0]
# for i in range(0, len(path_ins)):
#     file_processor(path_ins[i], path_outs[i], fps[i])