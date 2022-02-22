import os, shutil
import numpy as np
import math as m
import datetime as d
import random as r
from collections import namedtuple

from rtree import index
import h5py

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
        self.shift_x, self.shift_y, self.shift_z = (-1*reach.min for reach in self.robot.reach)
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
                                for p in position:
                                    p = pos_alignment(p)

                                # storing pos info
                                pos_info = (position, joint, vec_ee)
                                idx.insert(id, position[6].tolist(), obj=pos_info)

                                id += 1

        idx.close()

        end = d.datetime.now()
        print('done. duration: ', end-start)
        return filename

    def rtree_split(self, filename='raw_data'):
        foldername = RAW_DATA_FOLDER+filename
        ps.mkdir(foldername)
        if os.path.exists(foldername):
            print('dataset exists.')
            return 0

        start = d.datetime.now()

        size = [int((reach.max-reach.min)/self.diff)+1 for reach in self.robot.reach]

        id = np.zeros(size, dtype=int)

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
                                for p in position:
                                    p = pos_alignment(p)

                                # storing pos info
                                pos_info = (position, joint, vec_ee)
                                idx.insert(id, position[6].tolist(), obj=pos_info)

                                id += 1

        idx.close()

        end = d.datetime.now()
        print('done. duration: ', end-start)
        return filename

    def hdf5_store(self, filename='raw_data'):
        # self.filename = RAW_DATA_FOLDER+filename+'.npz'
        filename = RAW_DATA_FOLDER+filename+'.hdf5'
        start = d.datetime.now()

        with h5py.File(filename, 'a') as f:
            h5_data = f.create_group('franka_data')
            h5_data.attrs['scale'] = self.scale
            h5_data.attrs['shift'] = (self.shift_x, self.shift_y, self.shift_z)

            size = [int((reach.max-reach.min)/self.diff)+1 for reach in self.robot.reach]
            # dt = np.dtype([("pos_ee", np.float32, [3,]),\
            #                ("joint", np.float32, [7,]),\
            #                ("vec_ee", np.float32, [3,]),\
            #                ("index", np.uint32)])
            dt = np.dtype([("pos", np.float32, [7,3]),\
                           ("joint", np.float32, [7]),\
                           ("vec_ee", np.float32, [3])])

            pos_info = h5_data.create_dataset("pos_info", shape=size, dtype=h5py.vlen_dtype(dt))

            # index = 0
            # data_joints = []
            for j1 in range(int(self.joints[0].min*10), int(self.joints[0].max*10), int(self.scale*10)):
                for j2 in range(int(self.joints[1].min*10), int(self.joints[1].max*10), int(self.scale*10)):
                    for j3 in range(int(self.joints[2].min*10), int(self.joints[2].max*10), int(self.scale*10)):
                        for j4 in range(int(self.joints[3].min*10), int(self.joints[3].max*10), int(self.scale*10)):
                            for j5 in range(int(self.joints[4].min*10), int(self.joints[4].max*10), int(self.scale*10)):
                                for j6 in range(int(self.joints[5].min*10), int(self.joints[5].max*10), int(self.scale*10)):
                                    joints = np.array([j1/10.0, j2/10.0, j3/10.0, j4/10.0, j5/10.0, j6/10.0, 0.0])

                                    # cal fk
                                    position, vec_ee = self.robot.fk_jo(joints)
                                    # print(position)
                                    for i, j in enumerate(position):
                                        # print(j)
                                        for p, n in enumerate(j):
                                            position[i][p] = round(n, 4)

                                    # storing pos info
                                    # tmp_info = np.array([(position[6], joints, vec_ee, index)], dtype=dt)
                                    tmp_info = np.array([(position, joints, vec_ee)], dtype=dt)

                                    x = int((position[6][0]+self.shift_x)/self.diff)
                                    y = int((position[6][1]+self.shift_y)/self.diff)
                                    z = int((position[6][2]+self.shift_z)/self.diff)

                                    # print(x,y,z)
                                    pos_info[x, y, z] = np.append(pos_info[x, y, z], tmp_info)

                                    # data_joints.append(joints)

            # data_joints = np.asarray(data_joints)

            # np.savez(filename, joints=data_joints, positions=data_positions)

            # h5_data.create_dataset("joints", data=data_joints)




        end = d.datetime.now()
        print('done. duration: ', end-start)
        return filename

def high_dense_gen(iter, id=0):
    start = d.datetime.now()
    from ikpy.chain import Chain
    import ikpy.utils.plot as plot_utils
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], last_link_vector=[0, 0, 0], active_links_mask=[False, True, True, True, True, True, True, True, False, False])
    robot = Robot()

    # id = 0
    property = index.Property(dimension=3)
    idx = index.Index(RAW_DATA_FOLDER+'dense_'+str(iter), properties=property)

    # print([item.object for item in idx.nearest([0.0, 0.0, 0.2], 1, objects=True)])

    for i in range(iter):
        s = d.datetime.now()
        q = np.zeros(7)
        for j in range(6):
            q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)
        print(i, q)

        for x in range(200, 250, 2):
            for y in range(450, 500, 2):
                for z in range(300, 350, 2):
                    target = [x/1000, y/1000, z/1000]
                    joint = chain.inverse_kinematics(target, initial_position=[0, *q, 0, 0])[1:8]

                    position, vec_ee = robot.fk_jo(joint)
                    for p in position:
                        p = pos_alignment(p)

                    pos_info = (position, joint, vec_ee)
                    idx.insert(id, position[6].tolist(), obj=pos_info)

                    id += 1
        print(d.datetime.now()-s)
        if (i+1)%100 == 0 and not (i+1 == iter):
            idx.close()
            shutil.copyfile(RAW_DATA_FOLDER+'dense_'+str(iter)+'.idx', RAW_DATA_FOLDER+str(i+1)+'.idx')
            shutil.copyfile(RAW_DATA_FOLDER+'dense_'+str(iter)+'.dat', RAW_DATA_FOLDER+str(i+1)+'.dat')
            end = d.datetime.now()
            print(str(i+1)+' saved. duration: ', end-start)
            idx = index.Index(RAW_DATA_FOLDER+'dense_'+str(iter), properties=property)
    idx.close()
    end = d.datetime.now()
    print('done. duration: ', end-start)

if __name__ == '__main__':
    # dc = DataCollection(scale=30)
    # print(dc.hdf5_store('raw_data_7j_30'))

    # robot = Robot()
    # print(robot.fk_jo([0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]))

    # from multiprocessing import Process, Pool
    # pool = Pool()
    # pool.starmap(high_dense_gen, ((0, 100), (1562500, 100), (3125000, 100), (4687500, 100)))

    high_dense_gen(500)
