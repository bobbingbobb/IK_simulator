import sys
import numpy as np
import math as m
from collections import namedtuple
import os
import datetime as d
import random as r

from scipy.spatial import KDTree

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

raw_data = np.load('../data/raw_data/raw_data_7j_1.npz')
joints = raw_data['joints']
positions = raw_data['positions']

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


j = 0
s = d.datetime.now()
while j < 100000:
    j+=1
    # np.linalg.pinv(a)
    c = [i[0] for i in a]
    e = [i[2] for i in a]

m = d.datetime.now()

j = 0
while j < 100000:
    j+=1
    # np.linalg.inv(a)
    c=[]
    e=[]
    for i in a:
        c.append(i[0])
        e.append(i[2])
e = d.datetime.now()

s = m-s
e = e-m

# print(np.mean([s, e], axis=0))
# print(np.mean(np.array([]))

if [0,0,0]:
    print('aa')

from ik_simulator import IKTable, IKSimulator

# table = IKTable('raw_data_7j_1')

iks = IKSimulator()
for _ in range(50):
    x = round(r.uniform(-0.855, 0.855), 4)
    y = round(r.uniform(-0.855, 0.855), 4)
    z = round(r.uniform(-0.36, 1.19), 4)
    target = [x, y, z]
    # print(table.query_kd_tree(target))
    print(iks.find(target))
