import os
import numpy as np
import math as m
import datetime as d
from collections import namedtuple

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

class Gathering:
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
        self.table_name = self.__name_alignment(table_name)
        self.table = []

        self.raw_data = None
        self.joints = []
        self.positions = []

        self.shift_x, self.shift_y, self.shift_z = 0.855, 0.855, 0.36

        if os.path.exists(self.__tablename_alignment(self.table_name)):
            self.load_table()
        else:
            if raw_data == None:
                print('no such table.')
            else:
                print('creating table type with new raw data...')
                self.raw_data = self.__name_alignment(raw_data)
                self.load_data()
                self.create_table()

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

    def load_table(self):
        table_info = np.load(self.__tablename_alignment(self.table_name), allow_pickle=True)
        self.raw_data = str(table_info['raw_data'])
        self.table = table_info['table']

        print('table: ', self.table_name)
        print('data: ', self.raw_data)
        self.load_data()

    def create_table(self):
        start = d.datetime.now()
        self.table_v1()
        end = d.datetime.now()
        print('done. duration: ', end-start)

        self.load_table()

    def load_data(self):
        raw_data = np.load(self.__dataname_alignment(self.raw_data))
        self.joints = raw_data['joints']
        self.positions = raw_data['positions']

    def switch_raw_data(self, raw_data=None):
        if raw_data == 'empty':
            print('new raw_data needed.')
            return 0

        self.raw_data = self.__name_alignment(raw_data)
        self.load_data()
        print('switch to '+raw_data)

    def searching_area(self, target):
        for i,v in enumerate(target):
            target[i] = round(v, 4)

        target_space = self.searching_table_v1(target)

        return target_space

    def table_v1(self):
        #20 cm cube
        #x: -855 ~ 855, 1710, 18/20 = 9
        #y: -855 ~ 855, 1710, 18/20 = 9
        #z: -360 ~ 1190, 1550, 16/20 = 8
        grid_data = [[[[] for k in range(8)] for j in range(9)] for i in range(9)]
        for index, [_, _, _, _, _, _, [x, y, z]] in enumerate(self.positions):
            grid_data[int((x+self.shift_x)/0.2)][int((y+self.shift_y)/0.2)][int((z+self.shift_z)/0.2)].append(index)

        print('Density: ', self.__density(grid_data, 3))# avg sample in a 20cm cube

        np.savez(self.__tablename_alignment(self.table_name), raw_data=self.raw_data, table=grid_data)

    def searching_table_v1(self, target):
        searching_space = self.table[int((target[0]+self.shift_x)/0.2)][int((target[1]+self.shift_y)/0.2)][int((target[2]+self.shift_z)/0.2)]

        pos_jo = namedtuple('pos_jo', ['position', 'joint'])
        target_space = []
        for index in searching_space:
            target_space.append(pos_jo(self.positions[index][6], self.joints[index]))

        return target_space

class IKSimulator:
    def __init__(self):
        self.iktable = IKTable('table2')

    def find(self, target_position):
        searching_area = self.iktable.searching_area(target_position)

        return searching_area

if __name__ == '__main__':
    # gather = Gathering()
    # gather.without_colliding_detect('raw_data_7j_1')

    table2 = IKTable('table2', 'raw_data_7j_1')
    ik_simulator = IKSimulator()
    target = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]
    print(ik_simulator.find(target))
