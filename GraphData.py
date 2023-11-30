def GraphData(datalist, typeplotlist, labellist, Title, filename_data,\
              Axx = "x", Axy = "y",\
              left=None, right=None, bottom=None, top = None):
    from matplotlib import pyplot as plt
    #fig = plt.figure(figsize=(8, 8))
    #plt.rcParams['font.size'] = 16.
    ngraph = len(datalist)
    for il in range(ngraph):
        print(labellist[il])
        plt.plot(datalist[il][0], datalist[il][1], typeplotlist[il], 
                 label=labellist[il])
    plt.axis([left, right, bottom, top])
    if ngraph != 1:
        plt.legend(loc='best')
    plt.xlabel(Axx,fontsize=15)
    plt.ylabel(Axy,fontsize=15)
    plt.title(Title, fontsize=20, fontweight = 'bold', loc = 'center')
    plt.savefig(filename_data, bbox_inches='tight')
    #plt.grid()
    plt.show()
