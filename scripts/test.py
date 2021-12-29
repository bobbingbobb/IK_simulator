import sys
import numpy as np
import math as m
from collections import namedtuple
import os
import datetime as d
import random as r

from scipy.spatial import KDTree
from data_gen import Robot

from constants import *

import h5py

# list1 = np.array([[0.0, 0.0, 0.0]])
# list2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#
# list3 = np.array([[1.0, 1.0, 1.0]])
# list4 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#
# list = np.append(list1, list3, axis=0)
# print(list)



# np.savez('list', list)
# np.savez('list', list)
# kk = np.load('list.npz')
# print(kk)

c = 0
# for j1 in range(-28, 28, 5):
#     for j2 in range(-17, 17, 5):
#         for j3 in range(-28, 28, 5):
#             for j4 in range(-30, 0, 5):
#                 for j5 in range(-28, 28, 5):
#                     for j6 in range(0, 37, 5):
#                         c+=1

# data = np.load('data/franka_data.npz')
#
# print(data['joints'][0])
# print(data['positions'][0])
# print(data.key())

restrict = namedtuple('restrict', ['max', 'min'])
joint = [restrict(2.8973, -2.8973), restrict(1.7628, -1.7628), restrict(2.8973, -2.8973), restrict(0.0698, -3.0718), restrict(2.8973, -2.8973), restrict(3.7525, -0.0175), restrict(2.8973, -2.8973)]
# print(joint)

class ppp():
    def __init__(self):
        restrict = namedtuple('restrict', ['max', 'min'])
        self.aa = '123'

    def a(self, kk='vvv'):
        joint = [restrict(2.8973, -2.8973), restrict(1.7628, -1.7628), restrict(2.8973, -2.8973), restrict(0.0698, -3.0718), restrict(2.8973, -2.8973), restrict(3.7525, -0.0175), restrict(2.8973, -2.8973)]
        print(kk)
        print(joint)

# k = ppp()
# k.a()
# print(os.path.exists('data/raw_data.npz'))

#20 cm cube
#x: -855 ~ 855, 1710, 18/20 = 9
#y: -855 ~ 855, 1710, 18/20 = 9
#z: -360 ~ 1190, 1550, 16/20 = 8
# shift_x, shift_y, shift_z = 0.855, 0.855, 0.36
# grid_data = [[[[] for k in range(8)] for j in range(9)] for i in range(9)]
# for index, [_, _, _, _, _, _, [x, y, z]] in enumerate(positions):
#     # print(x, y, z)
#     grid_data[int((x+shift_x)/0.2)][int((y+shift_y)/0.2)][int((z+shift_z)/0.2)].append(index)

# print(np.mean(np.array([len(k) for i in grid_data for j in i for k in j if len(k) != 0])))# avg sample in a 20cm cube
# print(np.mean(np.array(grid_data).reshape((1, int(np.prod(np.array(grid_data).shape))))))
# print(np.prod(np.array(grid_data).shape))

# grid_data = [[[1],[2,3],[3,3]],[[1],[2,3],[3,3]]]

def recur(list, num):
    result = []
    # print(result)
    num -= 1
    for element in list:
        # print(len(element))
        if not num == 0:
            result += recur(element, num)
        else:
            if not len(element) == 0:
                result += [len(element)]

    return result

# print(np.mean(np.array(recur(grid_data, 3))))

# print(m.sqrt((0.001) ** 2 * 3))
# print([(m.pi/180.0)*(0.5**i) for i in range(5)])
#
# for i in range(5, -1, -1):
#     print([j for j in range(i, 6, 1)])

a = [[1,8,3],[3,3,3],[1,2,3],[4,4,4],[9,8,7],[3,3,3],[1,2,3],[5,5,5]]
c = [22.2,.22,-22.1]

# print(np.unique(a, return_index=True))
# print([float(k) for k in str(c)[1:-1].split(',')])
#
# x, y, z = c
# print(x)
# for i in a:
#     print(i)

b = {'[2 , 3 , 4]':2, '[1 , 2 , 3]':3}
for i, v in enumerate(a):
    if v == i:
        print('t')
else:
    print('f')

a = [1,2,3]
b = [-3,-4,-5]
# print(a[:1]+a[2:])

# print(np.mean([1,2,3], axis=0))
# print(np.subtract(a,b))

a = [[1,8,3],[3,3,3],[1,2,3]]
# a = [[1,3],[2,3],[3,3]]

# print(np.dot(a, np.linalg.inv(a)))
# print(np.dot(b, a))
#
# print((np.array(c)>5).any())
# print(np.absolute(c))


tree = KDTree(a, leafsize=2, balanced_tree=True)
print(tree.query_ball_point([3,3,2], 0.0003))

vectors = [[0.001017, 0.018661, 0.0], [0.006918, 0.005827, -0.013618], [0.002694, 0.015783, -0.003783], [0.001464, 0.002767, 0.013237], [-0.001235, -0.004304, -0.000281], [0.004531, -0.001224, -0.001164]]
vec = [[0.004531, -0.001224, -0.001164], [0.001017, 0.018661, 0.0], [0.001464, 0.002767, 0.013237]]
# 5, 0, 3
target = [0.5545, -0.0, 0.6245]

# a = np.dot(np.linalg.inv(vec), target)
# print(np.dot(vec, a))

t = []
for i in range(50):
    x = round(r.uniform(-0.855, 0.855), 4)
    y = round(r.uniform(-0.855, 0.855), 4)
    z = round(r.uniform(-0.36, 1.19), 4)
    t.append([x, y, z])
# print(t)

if [0,0,0]:
    print('aa')

# from ik_simulator import IKTable, IKSimulator
#
# # table = IKTable('raw_data_7j_1')
#
# iks = IKSimulator()
# for _ in range(50):
#     x = round(r.uniform(-0.855, 0.855), 4)
#     y = round(r.uniform(-0.855, 0.855), 4)
#     z = round(r.uniform(-0.36, 1.19), 4)
#     target = [x, y, z]
#     # print(table.query_kd_tree(target))
#     print(iks.find(target))

# raw_info = np.load('../data/raw_data/raw_data_7j_20.npz')
# print(raw_info['positions'][0][6])
# print(raw_info['joints'][0])
# positions = [p[6] for p in raw_info['positions']]
# print(len(positions))
# print(len(np.unique(positions, axis=0)))


print(m.floor(2.6))

# from data_gen import Robot
# work_joints = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
# robot = Robot()

# print(robot.fk_dh(work_joints))

print(np.random.randn(1, 4, 2))


# with h5py.File('test.hdf5', 'w') as f:
#     dt = np.dtype([\
#         ("pos", np.float32, [3,]),\
#         ("joint", np.float32, [7,]),\
#         ("ee", np.float32, [3,]),\
#         ("index", np.float32)])
#     # strst = h5py.vlen_dtype(dt)
#     # dt = np.dtype('float32', shape=[3,3])
#     strst = h5py.vlen_dtype(dt)
#     # dt = h5py.vlen_dtype(np.dtype('int32'))
#
#     pos_info = f.create_dataset("pos_info", shape=(2,3,4,), maxshape=(2,3,4), dtype=strst)
#     # pos_info = f.create_dataset("pos_info", shape=(1,4 ), dtype='i8')
#     # pos_info[0] = [1,2,3,4]
#     # print(pos_info)
#
#     pos_info.attrs['scale'] = 0
#
#
#     v1 = np.array([([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0], 3.0)], dtype=dt)
#     v2 = np.array([([0.0, 2.0, 0.0], [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], [2.0, 0.0, 0.0], 3.0)], dtype=dt)
#     # v2 = np.array([['[0.1 0.0 0.0]','1'], ['[1.1 0.0 0.0 0.0 0.0 0.0 0.0]','1'], ['[2.1 0.0 0.0]','1'], ['[3.1]','1']], dtype=dt)
#     # c1 = np.array([0.0, 0.0, 0.0])
#     # c2 = np.array([0.1, 0.0, 0.0])
#     # c3 = np.array([0.2, 0.0, 0.0])
#     #
#     # v1 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=dt)
#     # v2 = np.array([[0.1, 0.0, 0.0], [2.1, 0.0, 0.0], [3.1, 0.0, 0.0]], dtype=dt)
#
#     # v1 =  [[np.array(['[0.0, 0.0, 0.0]'])],\
#     #         [np.array(['[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'])],\
#     #         [np.array(['[0.0, 0.0, 0.0]'])],\
#     #         [np.array(['[0.0]'])]]
#     # print(len(pos_info[0,0,0]))
#     pos_info[0,0,0] = np.append(pos_info[0,0,0], v1)
#     # pos_info[0,0,1] = v2
#
#     print(pos_info[0,0,0])
#     pos_info[0,0,0] = np.append(pos_info[0,0,0], v2)
#
#     print(pos_info[0,0,0])
#     print(pos_info[0,0,1])

# with h5py.File(RAW_DATA_FOLDER+'test.hdf5', 'r') as f:
#     f = f['franka_data']
#     print(f.attrs['shift'])
#     # print(f['pos_info'][14,6,22]['vec_ee'])
#     # print(f['pos_info'][:]['pos'][6])
#     print(f['pos_info'][:])

    # def table_v1(self):
    #     #20 cm cube
    #     #x: -855 ~ 855, 1710, 18/20 = 9
    #     #y: -855 ~ 855, 1710, 18/20 = 9
    #     #z: -360 ~ 1190, 1550, 16/20 = 8
    #     grid_data = [[[[] for k in range(8)] for j in range(9)] for i in range(9)]
    #     for index, [_, _, _, _, _, _, [x, y, z]] in enumerate(self.positions):
    #         grid_data[int((x+self.shift_x)/0.2)][int((y+self.shift_y)/0.2)][int((z+self.shift_z)/0.2)].append(index)
    #
    #     print('Density: ', self.__density(grid_data, 3))# avg sample in a 20cm cube
    #
    #     np.savez(self.__tablename_alignment(self.table_name), raw_data=self.raw_data, table=grid_data)
    #
    # def searching_table_v1(self, target):
    #     searching_space = self.table[int((target[0]+self.shift_x)/0.2)][int((target[1]+self.shift_y)/0.2)][int((target[2]+self.shift_z)/0.2)]
    #
    #     pos_jo = namedtuple('pos_jo', ['position', 'joint'])
    #     target_space = []
    #     for index in searching_space:
    #         target_space.append(pos_jo(self.positions[index][6], self.joints[index]))
    #
    #     return target_space

robot = Robot()

p1 = [ 0.5602, -0.001 ,  0.6294]
j1 = [-2.8, -1.7,  0.8, -1.2, -1.3,  0.6,  0. ]
p1a, v1ee = robot.fk_jo(j1)
p2 = [ 0.5539, -0.0049,  0.6228]
j2 = [-2.8, -0.8,  2.6, -0.9, -1.9,  0. ,  0. ]
p2a, v2ee = robot.fk_jo(j2)

v1ee/=np.linalg.norm(v1ee)
v2ee/=np.linalg.norm(v2ee)
# print(v1ee)
# print(v2ee)
print(np.dot(v1ee, v2ee))

print(np.linalg.norm(p1a[3]-p2a[3]))
print(np.linalg.norm(p1a[5]-p2a[5]))
j2_ = np.append(j1[:3], j2[3:])
p2a_, v2ee_ = robot.fk_jo(j2_)
# v2ee_/=np.linalg.norm(v2ee_)
# print(v2ee_)

print(np.linalg.norm(p1a[5]-p2a_[5]))
print(np.linalg.norm([0.316, 0.0825]))#j1 - j3 range
print(np.linalg.norm([0.384, 0.0825]))#j3 - j5 range
