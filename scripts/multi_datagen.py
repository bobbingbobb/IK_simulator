import os, shutil
import numpy as np
import math as m
import datetime as d
import random as r
from collections import namedtuple

from rtree import index
import copy as c

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


def pos_alignment(position):
    for i,v in enumerate(position):
        position[i] = round(v, 4)
    return position

def multi_collect(j1_range, filename='raw_data'):
    filename = 'data/'+filename
    if os.path.exists(filename+'idx'):
        print('dataset exists.')
        return 0
    start = d.datetime.now()

    robot = Robot()
    scale = 12
    id = 1
    joint = np.zeros(7)

    # rtree preparing
    p = index.Property(dimension=3, fill_factor=0.9)
    idx = index.Index(filename, properties=p)

    for j1 in j1_range:
        joint[0] = j1/10.0
        for j2 in range(int(robot.joints[1].min*10), int(robot.joints[1].max*10), int(scale*10)):
            joint[1] = j2/10.0
            for j3 in range(int(robot.joints[2].min*10), int(robot.joints[2].max*10), int(scale*10)):
                joint[2] = j3/10.0
                for j4 in range(int(robot.joints[3].min*10), int(robot.joints[3].max*10), int(scale*10)):
                    joint[3] = j4/10.0
                    for j5 in range(int(robot.joints[4].min*10), int(robot.joints[4].max*10), int(scale*10)):
                        joint[4] = j5/10.0
                        for j6 in range(int(robot.joints[5].min*10), int(robot.joints[5].max*10), int(scale*10)):
                            # joint = np.array([j1/10.0, j2/10.0, j3/10.0, j4/10.0, j5/10.0, j6/10.0, 0.0])
                            joint[5] = j6/10.0

                            # cal fk
                            position, vec_ee = robot.fk_jo(joint)
                            for p in position:
                                p = pos_alignment(p)

                            # storing pos info
                            pos_info = (position, joint, vec_ee)
                            idx.insert(id, position[6].tolist(), obj=pos_info)

                            id += 1
                        # idx.close()
                        # return 0
            # print(d.datetime.now()-start)
            idx.close()
            shutil.copyfile(filename+'.idx', filename+'_1'+'.idx')
            shutil.copyfile(filename+'.dat', filename+'_1'+'.dat')
            end = d.datetime.now()
            print(str(i+1)+' saved. duration: ', end-start)
            idx = index.Index(filename, properties=p)

    end = d.datetime.now()
    print('done. duration: ', end-start)
    return filename

def high_dense_gen(iter, name, xs, ys, zs):
    start = d.datetime.now()
    from ikpy.chain import Chain
    import ikpy.utils.plot as plot_utils
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    robot = Robot()

    id = 1
    property = index.Property(dimension=3, fill_factor=0.9)
    idx = index.Index('data/'+name+'_'+str(iter), properties=property)

    # print([item.object for item in idx.nearest([0.0, 0.0, 0.2], 1, objects=True)])

    for i in range(iter):
        s = d.datetime.now()
        q = np.zeros(7)
        # for j in range(6):
        #     q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)
        # print(i, q)
        target = [0.0, 0.0, 0.0]

        for x in range(xs, xs+50, 1):
            target[0] = x/1000
            for y in range(ys, ys+50, 1):
                target[1] = y/1000
                for z in range(zs, zs+50, 1):
                    error = True
                    target[2] = z/1000

                    while error:
                        for j in range(6):
                            q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)
                        try:
                            joint = chain.inverse_kinematics(target, initial_position=[0, *q, 0])[1:8]
                            # position, _ = robot.fk_jo(joint)
                            position, vec_ee = robot.fk_jo(joint)
                            if not target == [round(p, 4) for p in position[6]]:
                                # print(target, position[6])
                                continue

                            for p in position:
                                p = pos_alignment(p)
                            pos_info = (position, joint, vec_ee)
                            idx.insert(id, position[6].tolist(), obj=pos_info)
                            # idx.insert(id, c.copy(target), obj=joint)

                            id += 1
                            error = False
                        except ValueError:
                            print('Error raised')
                            continue


            print(d.datetime.now()-s)
        # if (i+1)%2 == 0 and not (i+1 == iter):
        idx.close()
        shutil.copyfile('data/'+name+'_'+str(iter)+'.idx', 'data/'+name+str(i+1)+'.idx')
        shutil.copyfile('data/'+name+'_'+str(iter)+'.dat', 'data/'+name+str(i+1)+'.dat')
        end = d.datetime.now()
        print(str(i+1)+' saved. duration: ', end-start)
        idx = index.Index('data/'+name+'_'+str(iter), properties=property)
    idx.close()
    end = d.datetime.now()
    print('done. duration: ', end-start)

if __name__ == '__main__':
    robot = Robot()

    from multiprocessing import Process, Pool
    pool = Pool()

    j1 = (robot.joints[0].max-robot.joints[0].min)/19
    j1_range = [round(robot.joints[0].min+j1*j, 2) for j in range(20)]

    work = []
    for i in range(0, 20, 2):
        work.append((j1_range[i:i+2], str(int(i/2))+'rtree_10'))
    print(tuple(work))

    pool.starmap(multi_collect, tuple(work))

    i = 4
    #50
    # pool.starmap(high_dense_gen, ((i, '0dense', 200, 450, 300), (i, '1dense', 250, 450, 300), (i, '2dense', 200, 500, 300), (i, '3dense', 250, 500, 300)))
    # pool.starmap(high_dense_gen, ((i, '4dense', 200, 450, 350), (i, '5dense', 250, 450, 350), (i, '6dense', 200, 500, 350), (i, '7dense', 250, 500, 350)))
