import os, time
import numpy as np
import random as r
import math as m
from collections import namedtuple

from ik_simulator import IKTable

ik_dict = {}

def diff_cal(list_1, list_2):
    if len(list_1) == len(list_2):
        return m.sqrt(sum([(i - j)**2 for i, j in zip(list_1, list_2)]))
    else:
        print('length of two lists must be equal')
        return 0

def rotate_z(angle:float):
    rz = np.array([[m.cos(angle), -m.sin(angle), 0.0, 0.0],\
                   [m.sin(angle), m.cos(angle), 0.0, 0.0],\
                   [0.0, 0.0, 1.0, 0.0],\
                   [0.0, 0.0, 0.0, 1.0]])
    return rz
def rotate_y(angle:float):
    ry = np.array([[m.cos(angle), 0.0, m.sin(angle), 0.0],\
                   [0.0, 1.0, 0.0, 0.0],\
                   [-m.sin(angle), 0.0, m.cos(angle), 0.0],\
                   [0.0, 0.0, 0.0, 1.0]])
    return ry
def rotate_x(angle:float):
    rx = np.array([[1.0, 0.0, 0.0, 0.0],\
                   [0.0, m.cos(angle), -m.sin(angle), 0.0],\
                   [0.0, m.sin(angle), m.cos(angle), 0.0],\
                   [0.0, 0.0, 0.0, 1.0]])
    return rx

def fk_jo(joints:list):
    # [x, y, z, angle of the joint]
    # jo = np.array([[    0.0,    0.0, 0.333,     0.0],\
    #                [    0.0,    0.0,   0.0, -m.pi/2],\
    #                [    0.0, -0.316,   0.0,  m.pi/2],\
    #                [ 0.0825,    0.0,   0.0,  m.pi/2],\
    #                [-0.0825,  0.384,   0.0, -m.pi/2],\
    #                [    0.0,    0.0,   0.0,  m.pi/2],\
    #                [  0.088,    0.0,   0.0,  m.pi/2]])
    # cam = np.array([ 0.0424, -0.0424, 0.14, m.pi/4])# angle: dep_img coord to last joint
    # gripper = np.array([ 0.0, 0.0, 0.107+0.0584+0.06, 0.0])
    # flange = np.array([ 0.0, 0.0, 0.107, 0.0])

    #show position of every joint
    jo = np.array([[    0.0,    0.0,   0.14,     0.0],\
                   [    0.0,    0.0,  0.193, -m.pi/2],\
                   [    0.0, -0.193,    0.0,  m.pi/2],\
                   [ 0.0825,    0.0,  0.123,  m.pi/2],\
                   [-0.0825, 0.1245,    0.0, -m.pi/2],\
                   [    0.0,    0.0, 0.2595,  m.pi/2],\
                   [  0.088, -0.107,    0.0,  m.pi/2]])


    fk_mat = np.eye(4)
    trans_mat = np.eye(4)

    #joints
    for i in range(7):
        for j in range(3):
            trans_mat[j,3] = jo[i,j]
        fk_mat = np.dot(fk_mat, trans_mat)
        fk_mat = np.dot(fk_mat, rotate_x(jo[i, 3]))
        fk_mat = np.dot(fk_mat, rotate_z(joints[i]))
        # print(fk_mat[:3,3].tolist())

    return fk_mat[:3,3].tolist()

def fk_dh(joints:list):
    # dh: joint(theta), distance between axes(-1 a), movement on axis(d), angle(-1 alpha)
    dh = np.array([[0.0,     0.0, 0.333,     0.0],\
                   [0.0,     0.0,   0.0, -m.pi/2],\
                   [0.0,     0.0, 0.316,  m.pi/2],\
                   [0.0,  0.0825,   0.0,  m.pi/2],\
                   [0.0, -0.0825, 0.384,  -m.pi/2],\
                   [0.0,     0.0,   0.0,  m.pi/2],\
                   [0.0,   0.088, 0.107,  m.pi/2]])

    dh[:,0] = joints
    mat = np.eye(4)
    for i in range(7):
        dh_t = [[m.cos(dh[i,0])               , -m.sin(dh[i,0])               ,  0             ,  dh[i,1]               ],\
        		[m.sin(dh[i,0])*m.cos(dh[i,3]),  m.cos(dh[i,0])*m.cos(dh[i,3]), -m.sin(dh[i,3]), -dh[i,2]*m.sin(dh[i,3])],\
        		[m.sin(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,3]),  dh[i,2]*m.cos(dh[i,3])],\
        		[0                            ,  0                            ,  0             ,  1                     ]]
        mat = np.dot(mat, dh_t)
        # print([p[3] for p in mat[:3]])
        # print(mat)
    return [p[3] for p in mat[:3]]

def linear_interpolation(joint_a, joint_b, pos_a, pos_b, pos_c):
    dist_a = m.sqrt(sum([(i - j)**2 for i, j in zip(pos_a, pos_c)]))
    dist_b = m.sqrt(sum([(i - j)**2 for i, j in zip(pos_b, pos_c)]))

    prop_a = dist_a/(dist_a+dist_b)
    print(prop_a)

    tmp_joint = [i - (i - j)* prop_a for i, j in zip(joint_a, joint_b)]
    tmp_pos = fk_dh(tmp_joint)
    print(tmp_pos)
    diff = diff_cal(tmp_pos, pos_c)
    print(diff)
    return tmp_joint

    # if abs(pre_diff - diff) <= 0.0000000001:
    #     return tmp_joint
    # else:
    #     return interpolation_A(joint_a, tmp_joint, pos_a, tmp_pos, pos_c, diff)

def dimension_portion(joint, target_pos):
    iktable = IKTable('table2')

    for i,v in enumerate(target_pos):
        target_pos[i] = round(v, 4)

    searching_space = iktable.table[int((target_pos[0]+0.855)/0.2)][int((target_pos[1]+0.855)/0.2)][int((target_pos[2]+0.36)/0.2)]
    pos_jo = namedtuple('pos_jo', ['position', 'joint'])
    target_space = []
    for index in searching_space:
        target_space.append(pos_jo(iktable.positions[index], iktable.joints[index]))

    #threshold
    # min = 10000000
    # max = 0
    # for j1 in target_space:
    #     for j2 in target_space:
    #         diff = diff_cal(j1.joint, j2.joint)
    #         if not diff == 0:
    #             min = diff if (min > diff) else min
    #             max = diff if (max < diff) else max
    # print(min)  #0.5
    # print(max)  #10

    #finding arm posture types
    threshold = 1.5
    posture = np.full(len(target_space), -1)
    posture[0] = 0
    for i_joint in range(1, len(target_space), 1):
        for i_type in np.unique(posture):
            if not i_type == -1:
                diff = diff_cal(target_space[i_type].joint, target_space[i_joint].joint)
                if diff < threshold:
                    posture[i_joint] = i_type
                    break
        posture[i_joint] = i_joint if posture[i_joint] == -1 else posture[i_joint]

    # print(np.unique(posture))
    #take a posture
    # for jo in range(len(posture)):
    #     if posture[jo] == 0:
    #         print(target_space[jo].joint)

    #moving joint:[6,5], [6,5,4], ..., [6,5,4,3,2,1,0]
    tmp_joint = target_space[0].joint
    for i in range(5, -1, -1):
        moving_joint = [j for j in range(i, 7, 1)]
        print(moving_joint)
        tmp_joint = approximation(tmp_joint, target_pos, moving_joint=approx_joint)

        if diff_cal(fk_dh(tmp_joint), target_pos) < 0.0001:
            break

    print(target_space[0].position[6])
    print(fk_dh(tmp_joint))
    print(target_pos)

def approximation(nearest_joint, target_pos, moving_joint=[i for i in range(7)]):
    rad_offset = [(m.pi/180.0)*(0.5**i) for i in range(5)]  #[1, 0.5, 0.25, ...] degree
    diff = diff_cal(fk_dh(nearest_joint), target_pos)
    print(diff)

    tmp_joint = nearest_joint

    for i in moving_joint:
        for offset in rad_offset:
            reverse = 0
            while reverse < 2:
                tmp_joint[i] += offset
                pre_diff = diff
                tmp_pos = fk_dh(tmp_joint)
                diff = diff_cal(tmp_pos, target_pos)
                print(diff)
                if diff >= pre_diff:
                    offset *= -1
                    reverse += 1

            tmp_joint[i] += offset
            print('joint %s done' %(i+1))
            # if diff < 0.0001: # 0.01cm
            #     break

    return tmp_joint

def test(nearest_joint, target_pos):
    for n in range(20):
        nearest_joint[0] += m.pi/180.0
        tmp_pos = fk_dh(nearest_joint)
        diff = diff_cal(tmp_pos, target_pos)
        print(diff)

if __name__ == '__main__':
    #[ 0.5545 0  0.7315]
    joint_a:list = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
    pos_a = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]

    joint_b:list = [0.1, 0.1, 0.1, -1.5, -0.05, 2.1, 0.9]

    # pos_a = fk_dh(joint_a)
    # pos_b = fk_dh(joint_b)
    # print(pos_a)
    # print(pos_b)
    # pos_c = [(i+j)/2 for i, j in zip(pos_a, pos_b)]
    # pos_c = pos_b

    # joint_c = linear_interpolation(joint_a, joint_b, pos_a, pos_b, pos_c)
    # joint_c = approximation(joint_a, pos_c)

    dimension_portion(joint_a, pos_a)
    # print(fk_jo(joint_a))
    # print(fk_dh(joint_a))
