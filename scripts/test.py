import sys
import numpy as np
import math as m
from collections import namedtuple
import os
import datetime as d
import random as r

from scipy.spatial import KDTree

from data_gen import Robot
from ik_simulator import IKSimulator

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

print(m.floor(2.6))


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


def rotationMatrixToEulerAngles(R):
    sy = m.sqrt(R[0,0] ** 2 +  R[1,0] ** 2)
    singular = sy < 1e-6
    if  not singular :
        x = m.atan2(R[2,1] , R[2,2])
        y = m.atan2(-R[2,0], sy)
        z = m.atan2(R[1,0], R[0,0])
    else :
        x = m.atan2(-R[1,2], R[1,1])
        y = m.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def posture_comparison(pj, robot):
    thres_3 = np.linalg.norm([0.316, 0.0825])/10.0#j1 - j3 range
    thres_5 = (thres_3 + np.linalg.norm([0.384, 0.0825]))/10.0#j3 - j5 range

    nearby_postures = []
    for i_pos in pj:
        for ind, i_type in enumerate(nearby_postures):
            _, vecp_ee = robot.fk_dh(i_pos[1])
            _, vect_ee = robot.fk_dh(i_type[0][1])
            # print(vecp_ee)
            # print(vect_ee)
            # print(np.dot(vecp_ee, vect_ee))
            # print(i_pos.position)
            # print(i_type[0].position)
            # print(i_pos.position[3])
            # print(i_type[0].position[3])

            if np.dot(vecp_ee, vect_ee) > 0.9 and np.linalg.norm(i_pos[0][3]-i_type[0][0][3]) < thres_3 and np.linalg.norm(i_pos[0][5]-i_type[0][0][5]) < thres_5:
                nearby_postures[ind].append(i_pos)
        else:
            nearby_postures.append([i_pos])

        print(i_pos[0][6])

    # print(len(indices), len(nearby_postures))

    return nearby_postures

def pc_eeonly(pj, robot, scale, nearby_postures):
    # thres_3 = np.linalg.norm([0.316, 0.0825])#j1 - j3 range
    # thres_5 = (thres_3 + np.linalg.norm([0.384, 0.0825]))#j3 - j5 range

    # nearby_postures = []
    vecp_ee = [-0.3552042, -0.28940691, 0.88886085]

    pos = []
    for i_pos in pj:
        _, vect_ee = robot.fk_dh(i_pos[1])
        if np.dot(vecp_ee, vect_ee) > scale:
           pos.append(i_pos)

    scale += 0.05
    nearby_postures.append(pos)

    if len(pos) > 1:
        nearby_postures = pc_eeonly(pos, robot, scale, nearby_postures)

    return nearby_postures

def pc_disonly(pj, robot, scale, nearby_postures):
    thres_3 = np.linalg.norm([0.316, 0.0825])#j1 - j3 range
    thres_5 = (thres_3 + np.linalg.norm([0.384, 0.0825]))#j3 - j5 range

    pos = []
    for i_pos in pj:
        if np.linalg.norm(pj[3][0][3]-i_pos[0][3]) < thres_3/scale and \
           np.linalg.norm(pj[3][0][5]-i_pos[0][5]) < thres_5/scale:
           pos.append(i_pos)

    scale /= 2.0
    nearby_postures.append(pos)

    if len(pos) > 1:
        nearby_postures = pc_eeonly(pos, robot, scale, nearby_postures)

    return nearby_postures


np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
robot = Robot()
work_joints:list = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
fk_mat, vee = robot.fk_dh([-0.7,  0.1,  0.8, -2.1, -2.8,  0.6,  0. ])
print(rotationMatrixToEulerAngles(fk_mat))
print(vee)

test = [[[ 0.    ,  0.    ,  0.14  ],
       [ 0.    ,  0.    ,  0.333 ],
       [ 0.0147, -0.0124,  0.525 ],
       [ 0.106 , -0.0119,  0.6417],
       [ 0.2546,  0.0028,  0.6396],
       [ 0.4669,  0.0341,  0.4937],
       [ 0.5107, -0.0153,  0.6155]], [-0.7,  0.1,  0.8, -2.1, -2.8,  0.6,  0. ]]

# iks = IKSimulator()
# target = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]
# pj = iks.find(target)
# print(pj[3])
# # _, vecp_ee = robot.fk_dh(pj[3][1])
# # print(vecp_ee)
# print(len(pj))
# pc = pc_eeonly(pj, robot, 0.6, [])
# np.save('example_eeonly', np.array(pc, dtype=object), allow_pickle=True)
# pc = pc_disonly(pj, robot, 8.0, [])
# np.save('example_disonly', np.array(pc, dtype=object), allow_pickle=True)

# k = np.load('example_eeonly.npy', allow_pickle=True)
# print([len(kk) for kk in k])
# k = np.load('example_disonly.npy', allow_pickle=True)
# print([len(kk) for kk in k])



# jo_list = [j[1] for j in k[-2]]
# vec_ee = [-0.3552042, -0.28940691, 0.88886085]
# for j in jo_list:
#     _, vee = robot.fk_dh(j)
#     print(vee, np.dot(vec_ee, vee))


# pc = posture_comparison(pj[:1000], robot)
# print(len(pc))

# js = []
# for i,v in enumerate(pc):
#     if (len(v)>7):
#         print(i)
#         js.append([j[1] for j in v])
# print(js)
#
# np.save('js', js, allow_pickle=True)
# np.save('example', pc, allow_pickle=True)

# js = np.load('example.npy', allow_pickle=True)
# print(np.mean([len(j) for j in js]))

# ex = np.load('example.npy', allow_pickle=True)
# print(ex[12])
# for i,v in enumerate(ex):
#     if (len(v)>2):
#         print(i)

# print(posture_comparison(k[0], robot))

# p1 = [ 0.5602, -0.001 ,  0.6294]
# j1 = [-2.8, -1.7,  0.8, -1.2, -1.3,  0.6,  0. ]
# p1a, v1ee = robot.fk_jo(j1)
# p2 = [ 0.5539, -0.0049,  0.6228]
# j2 = [-2.8, -0.8,  2.6, -0.9, -1.9,  0. ,  0. ]
# p2a, v2ee = robot.fk_jo(j2)
#
# v1ee/=np.linalg.norm(v1ee)
# v2ee/=np.linalg.norm(v2ee)
# # print(v1ee)
# # print(v2ee)
# print(np.dot(v1ee, v2ee))
#
# print(np.linalg.norm(p1a[3]-p2a[3]))
# print(np.linalg.norm(p1a[5]-p2a[5]))
# j2_ = np.append(j1[:3], j2[3:])
# p2a_, v2ee_ = robot.fk_jo(j2_)
# # v2ee_/=np.linalg.norm(v2ee_)
# # print(v2ee_)
#
# print(np.linalg.norm(p1a[5]-p2a_[5]))
# print(np.linalg.norm([0.316, 0.0825]))#j1 - j3 range
# print(np.linalg.norm([0.384, 0.0825]))#j3 - j5 range

from rtree import index
p = index.Property()
p.dimension = 3
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index('rtree', properties=p)

idx.insert(1, (0.6, 0 ,0.3))
idx.insert(2, (0.5, 0 ,0.3))
idx.insert(3, (0.4, 0 ,0.3))
idx.insert(4, (0.3, 0 ,0.3))
idx.insert(5, (0.2, 0 ,0.3))

print(list(idx.nearest((0.4, 0 ,0.3), 3)))
