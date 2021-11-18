import numpy as np
import math as m
import datetime as d
from collections import namedtuple

DATA_FOLDER = '/data'
RAW_DATA_FOLDER = DATA_FOLDER+'/raw_data'
TABLE_FOLDER = DATA_FOLDER+'/table'

class Robot:
    def __init__(self):
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
        joint_num = 7

        fk_mat = np.eye(4)
        for i in range(joint_num):
            dh_mat = [[m.cos(self.dh[i,0])                    , -m.sin(self.dh[i,0])                    ,  0                  ,  self.dh[i,1]                    ],\
            		  [m.sin(self.dh[i,0])*m.cos(self.dh[i,3]),  m.cos(self.dh[i,0])*m.cos(self.dh[i,3]), -m.sin(self.dh[i,3]), -self.dh[i,2]*m.sin(self.dh[i,3])],\
            		  [m.sin(self.dh[i,0])*m.sin(self.dh[i,3]),  m.cos(self.dh[i,0])*m.sin(self.dh[i,3]),  m.cos(self.dh[i,3]),  self.dh[i,2]*m.cos(self.dh[i,3])],\
            		  [0                                      ,  0                                      ,  0                  ,  1                               ]]
            fk_mat = np.dot(fk_mat, dh_mat)
            # print(fk_mat[:3,3])

        return fk_mat[:3,3].tolist()

class Gathering:
    def __init__(self):
        self.robot = Robot()
        self.joints = robot.joints
        self.scale = 30 * m.pi/180
        # self.filename = RAW_DATA_FOLDER+'/raw_data.npz'

    def without_colliding_detect(self, scale=30, filename='raw_data'):
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
                                position = self.robot.fk(joints)
                                for i,v in enumerate(position):
                                    position[i] = round(v, 4)

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
    def __init__(self, raw_data = RAW_DATA_FOLDER+'/raw_data.npz'):
        self.raw_data = self.__filename_alignment(raw_data)
        self.joints = []
        self.positions = []

        self.shift_x, self.shift_y, self.shift_z = 0.855, 0.855, 0.36

        self.load_data()

    def __filename_alignment(self, filename):
        filename = filename.split('/')
        filename = filename[-1].split('.')
        filename = RAW_DATA_FOLDER+filename[0]+'.npz'
        return filename

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

    def load_data(self):
        raw_data = np.load(self.raw_data)
        self.joints = raw_data['joints']
        self.positions = raw_data['positions']

    def switch_raw_data(self, raw_data='empty'):
        if raw_data == 'empty':
            print('raw_data needed.')
            return 0

        self.raw_data = self.__filename_alignment(raw_data)
        load_data()
        print('switch to '+raw_data)

    def table_1(self, rebuild = False):
        #20 cm cube
        #x: -855 ~ 855, 1710, 18/20 = 9
        #y: -855 ~ 855, 1710, 18/20 = 9
        #z: -360 ~ 1190, 1550, 16/20 = 8
        table_name = TABLE_FOLDER+'table_1.npy'
        if rebuild:
            start = d.datetime.now()
            grid_data = [[[[] for k in range(8)] for j in range(9)] for i in range(9)]
            for index, [x, y, z] in enumerate(self.positions):
                grid_data[int((x+self.shift_x)/0.2)][int((y+self.shift_y)/0.2)][int((z+self.shift_z)/0.2)].append(index)
            end = d.datetime.now()
            print('done. duration: ', end-start)
            print(self.__density(grid_data, 3))# avg sample in a 20cm cube

            np.save(table_name, grid_data)
        else:
            grid_data = np.load(table_name)

        return (grid_data)

class IKSimulator:
    def __init__(self):
        pass

if __name__ == '__main__':
    ik = IKTable()
    ik.table_1()
