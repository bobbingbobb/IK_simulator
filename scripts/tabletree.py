import os, h5py
# import pyopencl as cl
import numpy as np
import pickle
# import boxtree
import math as m
import datetime as d

from constants import *

'''
class Graph:
    def __init__(self, dimension):
        self.quantity = None
        self.vertex = None
        self.edge = None
'''

class Node:
    def __init__(self):
        self.parent = None
        self.childern = None
        self.coordinate = None

class Leaf(Node):
    def __init__(self):
        super().__init__()

    def insert(self, position, object):
        pass


class IKTable:
    def __init__(self, filename):
        name = name_alignment(filename)
        self.table = None
        self.data = None

        if os.path.exists(TABLE_FOLDER+name+'.idx'):
            with open(TABLE_FOLDER+name+'.idx', 'wb') as f:
                self.table = pickle.load(f)

            f = h5py.File(RAW_DATA_FOLDER+name+'.hdf5', 'a')
            self.data = f['data']



    def create_dataset(self, scale, shift):
        self.data = f.create_group('data')

        self.data.attrs['scale'] = scale
        self.data.attrs['shift'] = shift
        self.data.create_dataset("pos", shape=size, dtype=h5py.vlen_dtype(dt))
        self.data.create_dataset("joint", shape=size, dtype=h5py.vlen_dtype(dt))
        self.data.create_dataset("vec_ee", shape=size, dtype=h5py.vlen_dtype(dt))

    def insert(self):
        pass

    def data_type(self, joint_num):
        dt = np.dtype([("pos", np.float32, [joint_num,3]),\
                       ("joint", np.float32, [joint_num]),\
                       ("vec_ee", np.float32, [3])])
        return dt

    def split(self):
        pass

    def delete(self):
        pass



# platform = cl.get_platforms()
# gpu = platform[0].get_devices(device_type=cl.device_type.GPU)
# ctx = cl.Context(devices=gpu)
# # ctx = cl.create_some_context()
# queue = cl.CommandQueue(ctx)
#
# positions = np.array([[1,2,3],[1,3,4],[2,3,5],[2,3,6],[2,3,7],[2,3,8],[2,3,9]])
#
# from boxtree import TreeBuilder
# tb = TreeBuilder(ctx)
# tree, _ = tb(queue, positions, max_particles_in_box=3)
#
# aq = boxtree.area_query.AreaQueryBuilder(ctx)
#
# print(ad(queue, tree, [2,3,6], 2))
