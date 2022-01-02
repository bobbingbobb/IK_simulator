import os, h5py
import numpy as np
import math as m
import datetime as d

from constants import *

class IKTable:
    def __init__(self, raw_data):
        self.table = []

        self.raw_data = RAW_DATA_FOLDER+self.__name_alignment(raw_data)+'.hdf5'
        self.joints = []
        self.all_posi = []
        self.vec_ee = []

        self.positions = []

        # self.pos_table = [] #dict: position to joint index    #no need

        self.shift_x, self.shift_y, self.shift_z = 0.855, 0.855, 0.36

        self.load_data()
        s = d.datetime.now()
        self.kd_tree()
        print(d.datetime.now()-s)


    def __name_alignment(self, name):
        name = str(name).split('/')
        name = name[-1].split('.')
        return name[0]

    def __density(self, data, data_dim):
        def recur(list, dim):
            result = []
            dim -= 1
            for element in list:
                # print(len(element))
                if not dim == 0:
                    result += recur(element, dim)
                else:
                    if not len(element) == 0:
                        result += [len(element)]

            return result

        return np.mean(np.array(recur(data, data_dim)))

    def __str2trans(self, key_str):
        return [float(k) for k in str(key_str)[1:-1].split(' ')]

    def load_data(self):
        s = d.datetime.now()
        print('loading data...')

        with h5py.File(self.raw_data, 'r') as f:
            f = f['franka_data']
            self.shift_x, self.shift_y, self.shift_z = f.attrs['shift']
            self.pos_info = f['pos_info']
            self.positions = [p['pos'][6] for p in self.pos_info]

        # raw_info = np.load(self.raw_data)
        # self.joints = raw_info['joints']
        # self.positions = [p[6] for p in raw_info['positions']]
        # self.all_posi = raw_info['positions']
        # print(len(self.all_posi))

        print('loading done. duration: ', d.datetime.now()-s)

    def switch_raw_data(self, raw_data=None):
        if raw_data == 'empty':
            print('new raw_data needed.')
            return 0

        self.raw_data = RAW_DATA_FOLDER+self.__name_alignment(raw_data)+'.hdf5'
        self.load_data()
        self.kd_tree()
        print('switch to '+raw_data)

    def searching_area(self, target):
        # return a list of position indices
        for i,v in enumerate(target):
            target[i] = round(v, 4)

        target_space = self.query_kd_tree(target)

        return target_space

    def kd_tree(self):
        self.table = KDTree(self.positions, leafsize=2, balanced_tree=True)

    def query_kd_tree(self, target, range = 0.05):

        result = self.table.query_ball_point(target, range)
        # print(range)

        # result = self.table.query(target, k=20, distance_upper_bound=0.05)[1]
        # result = np.setdiff1d(result, len(self.positions))

        # if (len(result) < 2):
        #     result = self.table.query(target, k=2)

        return result
