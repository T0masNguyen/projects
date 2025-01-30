import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import threading

class scatter_plot(threading.Thread):
    def __init__(selfself, flm_list, aru_list):
        X_flm = []
        Y_flm = []
        Z_flm = []
        color_flm = []

        # X_aru = []
        # Y_aru = []
        # Z_aru = []

        # hardcoded for 3 cameras
        i = 0
        landmarklist = []
        for cam_items in flm_list:  # for each camera
            for landmark in cam_items:
                item = [landmark, i]
                landmarklist.append(item)
            i = i + 1

        for item in landmarklist:
            x_ = []
            y_ = []
            z_ = []
            x_.append(item[1][3])
            y_.append(item[1][4])
            z_.append(item[1][5])

            x = 0
            y = 0
            z = 0
            counter = len(x_)
            for it in x_:
                x += it
            for it in y_:
                y += it
            for it in z_:
                z += it

            x = x / counter
            y = y / counter
            z = z / counter
            xx = 0
            yy = 0
            zz = 0

            for it in x_:
                xx += (it - x) * (it - x)
            sx = math.sqrt(xx / counter)
            for it in y_:
                yy += (it - y) * (it - y)
            sy = math.sqrt(yy / counter)
            for it in z_:
                zz += (it - z) * (it - z)
            sz = math.sqrt(zz / counter)

            X_flm.append(x)
            Y_flm.append(y)
            Z_flm.append(z)

            # colors depending on standard_deviation of - ToDO: Punktlagefehler
            if sx < 0.5:
                color_flm.append('green')
            elif sx > 1.005:
                color_flm.append('red')
            else:
                color_flm.append('orange')

        fig = plt.subplot() #  .figure(figsize=(8, 6))
        axes = plt.axes(projection="3d")
        # axes.scatter3D(x,y,z,color="red")

        axes.scatter3D(X_flm, Y_flm, Z_flm, color=color_flm, marker="o")  # circle marker for flm
        # axes.scatter3D(X_aru, Y_aru, Z_aru, color="blue", marker = "s") #square marker for aruco
        axes.set_title("3d facial landmarks precision", fontsize=14, fontweight="bold")
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Z")
        plt.tight_layout()


        print("\tscatter plot is runnung")

    def run(self):
        # plt.show()
        pass

def scatter_plot(flm_list, aru_list):
    '''
            Be careful ! hardcoded source code for 3 cameras.
            function instead: scatter_plot_dynamic

            Function to plot 3D- coordinates of the different cameras in one plot.

            :param flm_list: list with facial landmarks
            :param aru_list: list with arucos (to be implemented)

    '''
    X_flm = []
    Y_flm = []
    Z_flm = []
    color_flm = []

    # X_aru = []
    # Y_aru = []
    # Z_aru = []

    #hardcoded for 3 cameras
    i = 0
    landmarklist = []
    for cam_items in flm_list:   #for each camera
        for landmark in cam_items:
            item = [landmark, i]
            landmarklist.append(item)
        i = i+1



    for item in landmarklist:
        x_ = []
        y_ = []
        z_ = []
        x_.append(item[1][3])
        y_.append(item[1][4])
        z_.append(item[1][5])


        x=0
        y=0
        z=0
        counter = len(x_)
        for it in x_:
            x += it
        for it in y_:
            y += it
        for it in z_:
            z += it

        x = x / counter
        y = y / counter
        z = z / counter
        xx = 0
        yy = 0
        zz = 0

        for it in x_:
            xx += (it - x) * (it - x)
        sx = math.sqrt(xx / counter)
        for it in y_:
            yy += (it - y) * (it - y)
        sy = math.sqrt(yy / counter)
        for it in z_:
            zz += (it - z) * (it - z)
        sz = math.sqrt(zz / counter)

        X_flm.append(x)
        Y_flm.append(y)
        Z_flm.append(z)

        #colors depending on standard_deviation of - ToDO: Punktlagefehler
        if sx < 0.5:
            color_flm.append('green')
        elif sx > 1.005:
            color_flm.append('red')
        else:
            color_flm.append('orange')

    fig=plt.figure(figsize=(8,6))
    axes = plt.axes(projection="3d")
    #axes.scatter3D(x,y,z,color="red")

    axes.scatter3D(X_flm, Y_flm, Z_flm, color=color_flm, marker="o")  # circle marker for flm
   # axes.scatter3D(X_aru, Y_aru, Z_aru, color="blue", marker = "s") #square marker for aruco
    axes.set_title("3d facial landmarks precision",fontsize=14,fontweight="bold")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def scatter_plot_dynamic(flm_list, aru_list):
    '''
            Be careful ! Mistackes included.

            Function to plot 3D- coordinates of the different cameras in one plot.

            :param flm_list: list with facial landmarks
            :param aru_list: list with arucos (to be implemented)
    '''
    X_flm = []
    Y_flm = []
    Z_flm = []
    color_flm = []

    # X_aru = []
    # Y_aru = []
    # Z_aru = []

    # i = 0
    # for items in flm_list:   #for each camera
    #     if i == 0:
    #         #list_cam1.append(items)
    #         list_cam1 = items
    #     if i == 1:
    #         #list_cam2.append(items)
    #         list_cam2 = items
    #     if i == 2:
    #         # list_cam2.append(items)
    #         list_cam3 = items
    #     i = 1+i

    # i = 0
    # for items in aru_list:  # for each camera
    #     if i == 0:
    #         # list_cam1.append(items)
    #         list_aru1 = items
    #     if i == 1:
    #         # list_cam2.append(items)
    #         list_aru2 = items
    #     i = 1 + i

    i=0
    for idy in flm_list:
        for item1 in flm_list[i]:
            x_ = []
            y_ = []
            z_ = []
            x_.append(item1[3])
            y_.append(item1[4])
            z_.append(item1[5])

            ###Abbruch nach der 2. Kamera, ToDo: Dynamic loop in dependencie of the number of cameras ?!?
            j=i+1
            k=0
            for idx in flm_list[j]:
                if item1[0] == idx[0]:
                    x_.append(idx[3])
                    y_.append(idx[4])
                    z_.append(idx[5])
                    break
            k=0
            j = j + 1


            # for item3 in items:
            #     if item1[0] == item3[0]:
            #         x_.append(item3[3])
            #         y_.append(item3[4])
            #         z_.append(item3[5])
            #         break

            x=0
            y=0
            z=0
            counter = len(x_)
            for it in x_:
                x += it
            for it in y_:
                y += it
            for it in z_:
                z += it

            x = x / counter
            y = y / counter
            z = z / counter
            xx = 0
            yy = 0
            zz = 0

            for it in x_:
                xx += (it - x) * (it - x)
            sx = math.sqrt(xx / counter)
            for it in y_:
                yy += (it - y) * (it - y)
            sy = math.sqrt(yy / counter)
            for it in z_:
                zz += (it - z) * (it - z)
            sz = math.sqrt(zz / counter)

            X_flm.append(x)
            Y_flm.append(y)
            Z_flm.append(z)

            if sx < 0.5:
                color_flm.append('green')
            elif sx > 1.005:
                color_flm.append('red')
            else:
                color_flm.append('orange')
            i = i + 1

        # for item1 in list_aru1:
        #     for item2 in list_aru2:
        #         if item1[1][0] == item2[1][0]:
        #             x = (item1[3] + item2[3]) / 2
        #             y = (item1[4] + item2[4]) / 2
        #             z = (item1[5] + item2[5]) / 2
        #             X_aru.append(x)
        #             Y_aru.append(y)
        #             Z_aru.append(z)



    fig=plt.figure(figsize=(8,6))
    axes = plt.axes(projection="3d")
    #axes.scatter3D(x,y,z,color="red")

    axes.scatter3D(X_flm, Y_flm, Z_flm, color=color_flm, marker="o")  # circle marker for flm
   # axes.scatter3D(X_aru, Y_aru, Z_aru, color="blue", marker = "s") #square marker for aruco
    axes.set_title("3d facial landmarks precision",fontsize=14,fontweight="bold")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

