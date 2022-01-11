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



class RTree:
    class rectangle:
        def __init__(self, area):
            # assert isinstance(area, list)
            minx, miny, minz, maxx, maxy, maxz = area
            self.range_x = [minx, maxx]
            self.range_y = [miny, maxy]
            self.range_z = [minz, maxz]

    class node:
        def _create(root=False):
            if root:
                return RTree.node()

        def __init__(self):
            self.parent = None
            self.children = []

            self.coordinate = self.coordinate()

        def coordinate(self):
            if not self.children:
                return RTree.rectangle([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            for r in self.children:
                pass


        def __lt__(self, other):
            return id(self) < id(other)

        def __gt__(self, other):
            return id(self) > id(other)

        def __le__(self, other):
            return id(self) <= id(other)

        def __ge__(self, other):
            return id(self) >= id(other)

        def __eq__(self, other):
            return id(self) == id(other)

    class leaf(node):
        def __init__(self):
            super().__init__()

        def insert(self, position, object):
            pass

    class inner(node):
        def __init__(self):
            super().__init__()

        def insert(self, position, object):
            pass

    def create_tree(self):
        return RTree.node._create(root=True)

    def __init__(self, name):
        self.filename = TABLE_FOLDER+name+'.idx'
        self.size = 0
        self.index = 0

        # self.table = None
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                self.tree = pickle.load(f)
        else:
            self.tree = self.create_tree()


        # self.usable = []

    def insert(self, ):
        pass

class HDF5Data:
    def __init__(self, name):
        filename = RAW_DATA_FOLDER+name+'.hdf5'
        self.dt = self._data_type

        if not os.path.exists(filename):
            self.create_dataset(filename)

        f = h5py.File(filename, 'a')
        self.data = f['data']



    def create_dataset(self, filename):
        with h5py.File(filename, 'w'):
            self.data = f.create_group('data')

            # self.data.attrs['scale'] = scale
            # self.data.attrs['shift'] = shift
            self.data.create_dataset("pos", shape=(1,), dtype=h5py.vlen_dtype(dt['pos']))
            self.data.create_dataset("joint", shape=(1,), dtype=h5py.vlen_dtype(dt['joint']))
            self.data.create_dataset("vec_ee", shape=(1,), dtype=h5py.vlen_dtype(dt['vec_ee']))

    def _data_type(self):
        # dt = np.dtype([("pos", np.float32, [joint_num,3]),\
        #                ("joint", np.float32, [joint_num]),\
        #                ("vec_ee", np.float32, [3])])
        # return dt
        dt_joint = np.dtype([('j1', np.float32), ('j2', np.float32), ('j3', np.float32), ('j4', np.float32), ('j5', np.float32), ('j6', np.float32), ('j7', np.float32)])
        dt_vec = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        dt_pos = np.dtype([("j1", dt_vec),\
                           ("j2", dt_vec),\
                           ("j3", dt_vec),\
                           ("j4", dt_vec),\
                           ("j5", dt_vec),\
                           ("j6", dt_vec),\
                           ("j7", dt_vec)])

        dt = {}
        dt['pos'] = dt_pos
        dt['joint'] = dt_joint
        dt['vec_ee'] = dt_vec
        return dt

    def insert(self, pos, joint, vec_ee):
        pass

class IKTable:
    def __init__(self, filename):
        name = name_alignment(filename)
        self.table = TreeTable(name)
        self.data = HDF5Data(name)


        if os.path.exists(TABLE_FOLDER+name+'.idx'):
            with open(TABLE_FOLDER+name+'.idx', 'wb') as f:
                self.table = pickle.load(f)

            f = h5py.File(RAW_DATA_FOLDER+name+'.hdf5', 'a')
            self.data = f['data']

    def insert(self):
        pass

    def split(self):
        pass

    def delete(self):
        pass


if __name__ == '__main__':
    # table = IKTable('franka')
    tree = RTree('tree')
    print(tree)
    print(tree.size)
    print(tree.index)
    print(tree.tree)
    print(tree.tree.parent)
    print(tree.tree.children)
    print(tree.tree.coordinate)
