import numpy as np
import math as m
import datetime as d

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

def fk(joints:list):
    dh = np.array([[0.0,     0.0, 0.333,     0.0],\
                   [0.0,     0.0,   0.0, -m.pi/2],\
                   [0.0,     0.0, 0.316,  m.pi/2],\
                   [0.0,  0.0825,   0.0,  m.pi/2],\
                   [0.0, -0.0825, 0.384,  -m.pi/2],\
                   [0.0,     0.0,   0.0,  m.pi/2],\
                   [0.0,   0.088,   0.107,  m.pi/2]])
    # gripper = np.array([ 0.0, 0.0, 0.107+0.0584+0.06, 0.0])
    # flange = np.array([ 0.0, 0.0, 0.107, 0.0])

    dh[:,0] = joints
    joint_num = 7

    fk_mat = np.eye(4)
    for i in range(joint_num):
        dh_t = [[m.cos(dh[i,0])               , -m.sin(dh[i,0])               ,  0             ,  dh[i,1]               ],\
        		[m.sin(dh[i,0])*m.cos(dh[i,3]),  m.cos(dh[i,0])*m.cos(dh[i,3]), -m.sin(dh[i,3]), -dh[i,2]*m.sin(dh[i,3])],\
        		[m.sin(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,3]),  dh[i,2]*m.cos(dh[i,3])],\
        		[0                            ,  0                            ,  0             ,  1                     ]]
        fk_mat = np.dot(fk_mat, dh_t)
        print(fk_mat[:3,3])

    return fk_mat[:3,3].tolist()

def gather_data_without_colliding_detect():
    data_joints = []
    data_positions = []
    scale = int(0.5*10) #30 degrees,  duration=0:01:18.164633
    for j1 in range(-28, 28, scale):
        for j2 in range(-17, 17, scale):
            for j3 in range(-28, 28, scale):
                for j4 in range(-30, 0, scale):
                    for j5 in range(-28, 28, scale):
                        for j6 in range(0, 37, 5):
                            joints = [j1/10.0, j2/10.0, j3/10.0, j4/10.0, j5/10.0, j6/10.0, 0.0]
                            position = fk(joints)
                            for i,v in enumerate(position):
                                position[i] = round(v, 4)

                            data_joints.append(joints)
                            data_positions.append(position)

    data_joints = np.asarray(data_joints)
    data_positions = np.asarray(data_positions)
    np.savez('data/raw_data1.npz', joints=data_joints, positions=data_positions)

def load_data():
    pass

def main():
    start = d.datetime.now()
    # gather_data_without_colliding_detect()
    print(fk([0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]))
    end = d.datetime.now()
    print('done. duration: ', end-start)

def test():
    raw_data = np.load('data/raw_data1.npz')
    joints = raw_data['joints']
    positions = raw_data['positions']

    #20 cm cube
    #x: -855 ~ 855, 1710, 18/20 = 9
    #y: -855 ~ 855, 1710, 18/20 = 9
    #z: -360 ~ 1190, 1550, 16/20 = 8
    shift_x, shift_y, shift_z = 0.855, 0.855, 0.36
    grid_data = [[[[] for k in range(8)] for j in range(9)] for i in range(9)]
    for index, [x, y, z] in enumerate(positions):
        # print(x, y, z)
        grid_data[int((x+shift_x)/0.2)][int((y+shift_y)/0.2)][int((z+shift_z)/0.2)].append(index)


    print(np.mean(np.array([len(k) for i in grid_data for j in i for k in j if len(k) != 0])))# avg sample in a 20cm cube
    print(np.mean([np.where(np.array(grid_data) == 0, NaN, np.array(grid_data))]))# avg sample in a 20cm cube
    # grid_data = np.asarray(grid_data)
    # np.save('data/grid_data,npy', grid_data)

    target = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]
    for i,v in enumerate(target):
        target[i] = round(v, 4)

    searching_space = grid_data[int((target[0]+shift_x)/0.2)][int((target[1]+shift_y)/0.2)][int((target[2]+shift_z)/0.2)]
    print(positions[[index for index in searching_space]])


if __name__ == '__main__':
    main()
