import os
import numpy as np
import math as m
import datetime as d
from collections import namedtuple, defaultdict

from scipy.spatial import KDTree

DATA_FOLDER = '../data/'
RAW_DATA_FOLDER = DATA_FOLDER+'raw_data/'
TABLE_FOLDER = DATA_FOLDER+'table/'

class Robot:
    def __init__(self):
        self.joint_num = 7
        restrict = namedtuple('restrict', ['max', 'min'])
        self.joints = [restrict(2.8973, -2.8973), restrict(1.7628, -1.7628), restrict(2.8973, -2.8973), restrict(0.0698, -3.0718), restrict(2.8973, -2.8973), restrict(3.7525, -0.0175), restrict(2.8973, -2.8973)]
        self.dh = np.array([[0.0,     0.0, 0.333,     0.0],\
                            [0.0,     0.0,   0.0, -m.pi/2],\
                            [0.0,     0.0, 0.316,  m.pi/2],\
                            [0.0,  0.0825,   0.0,  m.pi/2],\
                            [0.0, -0.0825, 0.384,  -m.pi/2],\
                            [0.0,     0.0,   0.0,  m.pi/2],\
                            [0.0,   0.088,   0.107,  m.pi/2]])

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

    def fk(self, joints:list):
        self.dh[:,0] = joints

        fk_mat = np.eye(4)
        for i in range(self.joint_num):
            dh_mat = [[m.cos(self.dh[i,0])                    , -m.sin(self.dh[i,0])                    ,  0                  ,  self.dh[i,1]                    ],\
            		  [m.sin(self.dh[i,0])*m.cos(self.dh[i,3]),  m.cos(self.dh[i,0])*m.cos(self.dh[i,3]), -m.sin(self.dh[i,3]), -self.dh[i,2]*m.sin(self.dh[i,3])],\
            		  [m.sin(self.dh[i,0])*m.sin(self.dh[i,3]),  m.cos(self.dh[i,0])*m.sin(self.dh[i,3]),  m.cos(self.dh[i,3]),  self.dh[i,2]*m.cos(self.dh[i,3])],\
            		  [0                                      ,  0                                      ,  0                  ,  1                               ]]
            fk_mat = np.dot(fk_mat, dh_mat)
            # print(fk_mat[:3,3])

        return fk_mat[:3,3].tolist()

    def fk_jo(self, joints:list):
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
            pos.append(fk_mat[:3,3].tolist())
            # print(fk_mat[:3,3].tolist())

        return pos


class DataCollection:
    def __init__(self):
        self.robot = Robot()
        self.joints = self.robot.joints
        self.scale = 30 * m.pi/180
        # self.filename = RAW_DATA_FOLDER+'raw_data.npz'

    def without_colliding_detect(self, filename='raw_data', scale=30, ):
        scale = scale * m.pi/180
        # self.filename = RAW_DATA_FOLDER+filename+'.npz'
        filename = RAW_DATA_FOLDER+filename+'.npz'
        start = d.datetime.now()

        data_joints = []
        data_positions = []
        for j1 in range(int(self.joints[0].min*10), int(self.joints[0].max*10), int(self.scale*10)):
            for j2 in range(int(self.joints[1].min*10), int(self.joints[1].max*10), int(self.scale*10)):
                for j3 in range(int(self.joints[2].min*10), int(self.joints[2].max*10), int(self.scale*10)):
                    for j4 in range(int(self.joints[3].min*10), int(self.joints[3].max*10), int(self.scale*10)):
                        for j5 in range(int(self.joints[4].min*10), int(self.joints[4].max*10), int(self.scale*10)):
                            for j6 in range(int(self.joints[5].min*10), int(self.joints[5].max*10), int(self.scale*10)):
                                joints = [j1/10.0, j2/10.0, j3/10.0, j4/10.0, j5/10.0, j6/10.0, 0.0]
                                # position = self.robot.fk(joints)
                                position = self.robot.fk_jo(joints)
                                for i, j in enumerate(position):
                                    for p, n in enumerate(j):
                                        position[i][p] = round(n, 4)

                                data_joints.append(joints)
                                data_positions.append(position)

        data_joints = np.asarray(data_joints)
        data_positions = np.asarray(data_positions)
        # np.savez(self.filename, joints=data_joints, positions=data_positions)
        np.savez(filename, joints=data_joints, positions=data_positions)

        end = d.datetime.now()
        print('done. duration: ', end-start)
        return filename


class IKTable:
    def __init__(self, table_name, raw_data=None):
        # self.table_name = self.__name_alignment(table_name)
        self.table = []

        self.raw_data = None
        self.joints = []    #list: origin joint data
        self.positions = []
        self.pos_table = [] #dict: position to joint index

        self.shift_x, self.shift_y, self.shift_z = 0.855, 0.855, 0.36

        self.load_data()
        self.kd_tree()

    def __name_alignment(self, name):
        name = str(name).split('/')
        name = name[-1].split('.')
        return name[0]

    def __dataname_alignment(self, name):
        return RAW_DATA_FOLDER+self.__name_alignment(name)+'.npz'

    def __tablename_alignment(self, name):
        return TABLE_FOLDER+self.__name_alignment(name)+'.npz'

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
        return [float(k) for k in str(key_str)[1:-1].split(',')]

    def load_data(self):
        print('loading data...')
        raw_info = np.load(self.__dataname_alignment(self.raw_data))
        self.joints = raw_info['joints']

        pos_jo = defaultdict(list)
        for index, pos in enumerate(raw_info['positions']):
            pos_jo[str(list(pos[6]))].append(index)

        self.pos_table = pos_jo
        self.positions = [self.__str2trans(k) for k in pos_jo.keys()]

        print('loading done.')

    def switch_raw_data(self, raw_data=None):
        if raw_data == 'empty':
            print('new raw_data needed.')
            return 0

        self.raw_data = self.__name_alignment(raw_data)
        self.load_data()
        self.kd_tree()
        print('switch to '+raw_data)

    def searching_area(self, target):
        for i,v in enumerate(target):
            target[i] = round(v, 4)

        target_space = self.query_kd_tree(target)

        return target_space

    def kd_tree(self):
        self.table = KDTree(self.positions, leafsize=2, balanced_tree=True)

    def query_kd_tree(self, target):
        searching_space = self.table.query_ball_point(target, 0.02)

        target_space = []
        for key in searching_space:
            target_space.append(self.positions[key])

        return target_space


class IKSimulator:
    def __init__(self):
        self.iktable = IKTable('table3')

    def find(self, target_position):
        searching_area = self.iktable.searching_area(target_position)

        return searching_area


if __name__ == '__main__':
    # gather = DataCollection()
    # gather.without_colliding_detect('raw_data_7j_1')

    # table2 = IKTable('table3', 'raw_data_7j_1')
    ik_simulator = IKSimulator()
    target = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]
    print(ik_simulator.find(target))
