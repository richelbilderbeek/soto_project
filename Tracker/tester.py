__author__ = 'pieter'

import sys
import ConfigParser
import json
import csv
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Tester:
    def __init__(self, config):

        self.video_folder = config.get('default', 'video_folder')
        self.video_files = json.loads(config.get('default', 'video_files'))
        self.number_of_objects = json.loads(config.get('default', 'number_of_objects'))

        self.fig, self.ax = plt.subplots()

        self.lines = [self.ax.plot([], [], lw=2)[0] for _ in range(self.number_of_objects)]
        self.ax.set_ylim(0, 300)
        self.ax.set_xlim(0, 800)
        #self.ax.grid()
        self.xdata = [collections.deque(maxlen=100) for _ in range(self.number_of_objects)]
        self.ydata = [collections.deque(maxlen=100) for _ in range(self.number_of_objects)]

    def _to_matrix(self, l, n):
        return [self._to_int(l[i:i+n]) for i in xrange(0, len(l), n)]

    @staticmethod
    def read_data_from_csv(f):
        try:
            reader = csv.reader(f)  # creates the reader object
            return [row for row in reader]
        except:
            print "Couldn't read file"
            return []

    @staticmethod
    def _to_int(l):
        if isinstance(l, list):
            return [int(el) for el in l]
        return int(l)

    def order_data(self, data):
        return [self._to_matrix(x[3:3+self.number_of_objects*2], 2) for x in data[1:]]

    def get_data(self, video_file):
        self.output_file = self.video_folder + video_file[:video_file.find('.')]+'_output_corrected.csv'
        output_csv = open(self.output_file, "rb")
        data = self.read_data_from_csv(output_csv)
        output_csv.close()      # closing
        return self.order_data(data)

    def run(self):
        for video_file in self.video_files:
            data = self.get_data(video_file)

            ani = animation.FuncAnimation(self.fig, self.do, data, blit=True, interval=10, repeat=False)
            plt.show()


    def do(self, data):
        # update the data
        for idx, d in enumerate(data):
            x, y = d
            self.xdata[idx].append(x)
            self.ydata[idx].append(y)
            self.lines[idx].set_data(self.xdata[idx], self.ydata[idx])

        return self.lines

def main(args=None):
    """The main routine."""

    if not sys.argv[1:] is None:
        args = sys.argv[1:]

    config = ConfigParser.ConfigParser()

    result = config.read(args[0])

    if not result:
        print "Error({0}): {1}".format("couldn't find file", args[0])
        return 0

    tester = Tester(config)
    tester.run()

if __name__ == '__main__':
    main(['config.ini'])
