import os
import numpy as np
import math as m
import datetime as d
from collections import namedtuple
from rtree import index

from constants import *
from utilities import *

class Robot:
    def __init__(self):
        '''
        robot specification.

        #range
        #x: -855 ~ 855, 1710
        #y: -855 ~ 855, 1710
        #z: -360 ~ 1190, 1550
        '''
        self.joint_num = 7
        restrict = namedtuple('restrict', ['max', 'min'])
        self.reach = [restrict(0.855, -0.855), restrict(0.855, -0.855), restrict(1.19, -0.36)]
        self.joints = [restrict(2.8973, -2.8973), restrict(1.7628, -1.7628), restrict(2.8973, -2.8973), restrict(0.0698, -3.0718), restrict(2.8973, -2.8973), restrict(3.7525, -0.0175), restrict(2.8973, -2.8973)]
        self.dh = np.array([[0.0,     0.0, 0.333,     0.0],\
                            [0.0,     0.0,   0.0, -m.pi/2],\
                            [0.0,     0.0, 0.316,  m.pi/2],\
                            [0.0,  0.0825,   0.0,  m.pi/2],\
                            [0.0, -0.0825, 0.384, -m.pi/2],\
                            [0.0,     0.0,   0.0,  m.pi/2],\
                            [0.0,   0.088, 0.107,  m.pi/2]])

    def __rotate_z(self, angle:float):
        rz = np.array([[m.cos(angle), -m.sin(angle), 0.0, 0.0],\
                       [m.sin(angle), m.cos(angle), 0.0, 0.0],\
                       [0.0, 0.0, 1.0, 0.0],\
                       [0.0, 0.0, 0.0, 1.0]])
        return rz

    def __rotate_y(self, angle:float):
        ry = np.array([[m.cos(angle), 0.0, m.sin(angle), 0.0],\
                       [0.0, 1.0, 0.0, 0.0],\
                       [-m.sin(angle), 0.0, m.cos(angle), 0.0],\
                       [0.0, 0.0, 0.0, 1.0]])
        return ry

    def __rotate_x(self, angle:float):
        rx = np.array([[1.0, 0.0, 0.0, 0.0],\
                       [0.0, m.cos(angle), -m.sin(angle), 0.0],\
                       [0.0, m.sin(angle), m.cos(angle), 0.0],\
                       [0.0, 0.0, 0.0, 1.0]])
        return rx

    def fk_dh(self, joints:list):
        '''
        fk matrix with given dh.

        return:
            pos: end position
            vec_ee: end effector vector, presenting the facing of the end effector
                #todo: extend to customized ee offset
                #todo: fitting the euler angle presentation
        '''
        self.dh[:,0] = joints

        fk_mat = np.eye(4)
        for i in range(self.joint_num):
            dh_mat = [[m.cos(self.dh[i,0])                    , -m.sin(self.dh[i,0])                    ,  0                  ,  self.dh[i,1]                    ],\
            		  [m.sin(self.dh[i,0])*m.cos(self.dh[i,3]),  m.cos(self.dh[i,0])*m.cos(self.dh[i,3]), -m.sin(self.dh[i,3]), -self.dh[i,2]*m.sin(self.dh[i,3])],\
            		  [m.sin(self.dh[i,0])*m.sin(self.dh[i,3]),  m.cos(self.dh[i,0])*m.sin(self.dh[i,3]),  m.cos(self.dh[i,3]),  self.dh[i,2]*m.cos(self.dh[i,3])],\
            		  [0                                      ,  0                                      ,  0                  ,  1                               ]]
            fk_mat = np.dot(fk_mat, dh_mat)
            # print(fk_mat[:3,3])

        vec_z = [0.0, 0.0, 1.0, 0.0]
        vec_ee = np.dot(fk_mat, vec_z)

        return fk_mat[:3,3], vec_ee[:3]

    def fk_jo(self, joints:list):
        '''
        fk matrix with given specification of each joint.

        return:
            pos: positions of each joints
            vec_ee: end effector vector, presenting the facing of the end effector
                #todo: extend to customized ee offset
                #todo: fitting the euler angle presentation
        '''
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
        pos = []

        #joints
        for i in range(self.joint_num):
            for j in range(3):
                trans_mat[j,3] = jo[i,j]
            fk_mat = np.dot(fk_mat, trans_mat)
            fk_mat = np.dot(fk_mat, self.__rotate_x(jo[i, 3]))
            fk_mat = np.dot(fk_mat, self.__rotate_z(joints[i]))
            # pos = np.append(pos, fk_mat[:3,3], axis=0)
            pos.append(fk_mat[:3,3])
            # print(fk_mat[:3,3].tolist())

        vec_z = [0.0, 0.0, 1.0, 0.0]
        vec_ee = np.dot(fk_mat, vec_z)

        return np.array(pos), vec_ee[:3]


class DataCollection:
    def __init__(self, scale=30):
        '''
        collecting robot infomation.
        '''
        self.robot = Robot()
        self.joints = self.robot.joints
        self.diff = 0.05
        self.scale = scale * m.pi/180
        # self.shift_x, self.shift_y, self.shift_z = (-1*reach.min for reach in self.robot.reach)
        # self.filename = RAW_DATA_FOLDER+'raw_data.npz'

    def without_colliding_detect(self, filename='raw_data'):
        filename = RAW_DATA_FOLDER+filename
        if os.path.exists(filename+'idx'):
            print('dataset exists.')
            return 0
        start = d.datetime.now()

        id = 0

        # rtree preparing
        p = index.Property()
        p.dimension = 3
        idx = index.Index(filename, properties=p)

        for j1 in range(int(self.joints[0].min*10), int(self.joints[0].max*10), int(self.scale*10)):
            for j2 in range(int(self.joints[1].min*10), int(self.joints[1].max*10), int(self.scale*10)):
                for j3 in range(int(self.joints[2].min*10), int(self.joints[2].max*10), int(self.scale*10)):
                    for j4 in range(int(self.joints[3].min*10), int(self.joints[3].max*10), int(self.scale*10)):
                        for j5 in range(int(self.joints[4].min*10), int(self.joints[4].max*10), int(self.scale*10)):
                            for j6 in range(int(self.joints[5].min*10), int(self.joints[5].max*10), int(self.scale*10)):
                                joint = np.array([j1/10.0, j2/10.0, j3/10.0, j4/10.0, j5/10.0, j6/10.0, 0.0])

                                # cal fk
                                position, vec_ee = self.robot.fk_jo(joint)

                                # storing pos info
                                pos_info = (pos_alignment(position), joint, vec_ee)
                                idx.insert(id, position[6].tolist(), obj=pos_info)

                                id += 1

        idx.close()

        end = d.datetime.now()
        print('done. duration: ', end-start)
        return filename

if __name__ == '__main__':
    dc = DataCollection(scale=20)
    print(dc.without_colliding_detect('raw_data_7j_20'))
