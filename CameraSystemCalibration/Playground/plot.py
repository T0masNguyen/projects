import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import threading


class plot_3d(threading.Thread):
    def __init__(self):

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        plt.subplots_adjust(left=0.25, bottom=0.25)

        self.x = np.arange(0.0, 1.0, 0.1)
        a0 = 5
        b0 = 1
        y = a0 * self.x + b0
        z = np.zeros(10)

        self.l, = plt.plot(self.x, y, z)

        # Set size of Axes
        plt.axis([0, 1, -10, 10])

        # Place Sliders on Graph
        ax_a = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_b = plt.axes([0.25, 0.15, 0.65, 0.03])

        # Create Sliders & Determine Range
        self.sa = Slider(ax_a, 'a', 0, 10.0, valinit=a0)
        self.sb = Slider(ax_b, 'b', 0, 10.0, valinit=b0)

        self.sa.on_changed(self.update(10))
        self.sb.on_changed(self.update(10))

        plt.show()

    def run(self):
        self.update()

    def update(self, val):
        a = self.sa.val
        b = self.sb.val
        self.l.set_ydata(a*self.x+b)
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    plot = plot_3d()

    for i in range(1, 5):
        plot.update(i*10)

