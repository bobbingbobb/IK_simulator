import os, h5py
# import pyopencl as cl
import numpy as np
# import boxtree
import math as m
import datetime as d

from constants import *

class Graph:
    def __init__(self, dimension):
        self.quantity = None
        self.vertex = None
        self.edge = None


class Node:
    def __init__(self):
        self.parent = None
        self.childern = None
        self.bounding_box = None


class Leaf(Node):
    def __init__(self):
        super().__init__()

    def insert(self, position, object):
        pass


class IKTable:
    def __init__(self):
        pass

    def split(self):
        pass

    def insert(self):
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
