import os, time
import numpy as np
import random as r
import math as m
import datetime as d
from collections import namedtuple, defaultdict

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
    dist_a = np.linalg.norm(pos_a - pos_c)
    dist_b = np.linalg.norm(pos_b - pos_c)

    prop_a = dist_a/(dist_a+dist_b)
    print(prop_a)

    tmp_joint = [i - (i - j)* prop_a for i, j in zip(joint_a, joint_b)]
    tmp_pos = fk_dh(tmp_joint)
    print(tmp_pos)
    diff = np.linalg.norm(tmp_pos - pos_c)
    print(diff)
    return tmp_joint

    # if abs(pre_diff - diff) <= 0.0000000001:
    #     return tmp_joint
    # else:
    #     return interpolation_A(joint_a, tmp_joint, pos_a, tmp_pos, pos_c, diff)

def find_all_posture(joint, target_pos):
    start = d.datetime.now()

    iktable = IKTable('table2')

    for i,v in enumerate(target_pos):
        target_pos[i] = round(v, 4)

    searching_space = iktable.table[int((target_pos[0]+0.855)/0.2)][int((target_pos[1]+0.855)/0.2)][int((target_pos[2]+0.36)/0.2)]
    pos_jo = namedtuple('pos_jo', ['position', 'joint'])
    target_space = []
    for index in searching_space:
        target_space.append(pos_jo(iktable.positions[index], iktable.joints[index]))
    # print(len(target_space))

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
    find_posture = np.full(len(target_space), -1)
    find_posture[0] = 0
    for i_joint in range(1, len(target_space), 1):
        for i_type in np.unique(find_posture):
            if not i_type == -1:
                diff = diff_cal(target_space[i_type].joint, target_space[i_joint].joint)
                if diff < threshold:
                    find_posture[i_joint] = i_type
                    break
        find_posture[i_joint] = i_joint if find_posture[i_joint] == -1 else find_posture[i_joint]

    print(np.sort([diff_cal(target_space[i].position[6], target_pos) for i in np.unique(find_posture)]))

    # end = d.datetime.now()
    # print(end-start)
    # print(np.unique(find_posture))
    #take a find_posture
    # for jo in range(len(find_posture)):
    #     if find_posture[jo] == 0:
    #         print(target_space[jo].joint)

    #moving joint:[5,4], ..., [5,4,3,2,1,0], joint[6] does not affect position
    n = 0.0
    origin_diff = []
    time = []
    jo_diff = namedtuple('jo_diff', ['joint', 'diff'])
    posture = []
    for i_type in np.unique(find_posture):
        tmp_joint = target_space[i_type].joint
        moving_joint = [j for j in range(6)]

        for i in range(4, -1, -1):
            moving_joint = [j for j in range(i, 6, 1)]
            tmp_joint, diff, t = approximation(tmp_joint, target_pos, moving_joint=moving_joint)
            # tmp_joint, diff = approximation(tmp_joint, target_pos, moving_joint=moving_joint)

        posture.append(jo_diff(tmp_joint, diff))
        time.append(t)

        if diff > 0.005:#0.5cm
            n += 1
            origin_diff.append(diff_cal(target_space[i_type].position[6], target_pos))
            # origin_diff.append([round(m.sqrt((a-b)**2), 4) for a,b in zip(target_space[i_type].position[6], target_pos)])

    print(target_pos)
    # print(fk_dh(tmp_joint))
    print('mean diff: ', np.mean(np.array([p.diff for p in posture])))
    print(np.sort(origin_diff))
    print('worst: ',max([p.diff for p in posture]))
    print('worst%: ', n/len(posture))
    print(np.mean(np.array(time)))


    end = d.datetime.now()
    print(end-start)

def approximation(nearest_joint, target_pos, moving_joint=[i for i in range(7)]):
    start = d.datetime.now()

    rad_offset = [(m.pi/180.0)*(0.5**i) for i in range(3)]  #[1, 0.5, 0.25] degree
    diff = diff_cal(fk_dh(nearest_joint), target_pos)
    # print(diff)

    tmp_joint = nearest_joint

    for i in moving_joint:
        for offset in rad_offset:
            reverse = 0
            while reverse < 2:
                tmp_joint[i] += offset
                pre_diff = diff
                tmp_pos = fk_dh(tmp_joint)
                diff = diff_cal(tmp_pos, target_pos)
                # print(tmp_pos, diff)
                if diff >= pre_diff:
                    offset *= -1
                    reverse += 1

            tmp_joint[i] += offset
            # print('joint %s with %s done' %(i+1, offset))

    end = d.datetime.now()
    # print(end-start)

    return tmp_joint, pre_diff, end-start
    # return tmp_joint, pre_diff

def test(nearest_joint, target_pos):
    for n in range(20):
        nearest_joint[0] += m.pi/180.0
        tmp_pos = fk_dh(nearest_joint)
        diff = diff_cal(tmp_pos, target_pos)
        print(diff)

def str2trans(key_str):
    return [float(k) for k in str(key_str)[1:-1].split(' ')]

def load_data(raw_dataname):
    print('loading data...')
    raw_data = np.load(raw_dataname)
    joints = raw_data['joints']

    s = d.datetime.now()
    pos_jo = defaultdict(list)
    for index, pos in enumerate(raw_data['positions']):
        pos_jo[str(list(pos[6]))].append(index)
        # break
    e = d.datetime.now()
    print(e-s)

    pos_table = pos_jo
    # print(pos_jo.keys())
    positions1 = [[float(k) for k in str(a)[1:-1].split(',')] for a in pos_jo.keys()]
    m = d.datetime.now()
    print(m-e)

    positions = []
    pos_table = []
    for jo_ind, pos in enumerate(raw_data['positions']):
        print(jo_ind)
        for p_ind, p in enumerate(positions):
            # print(pos[6])
            # print(p)
            if (pos[6] == p).all():
                pos_table[p_ind].append(jo_ind)
        else:
            positions.append(pos[6])
            pos_table.append([])
            pos_table[-1].append(jo_ind)

    e = d.datetime.now()
    print(positions1==positions)
    print(m-s)
    print(e-m)

def sort():
    data = np.load('../data/raw_data/raw_data_7j_1.npz')

    position = data['positions']
    s = d.datetime.now()
    p = np.sort([i[6][0] for i in position])
    m = d.datetime.now()
    p = np.percentile([i[6][0] for i in position], 50)
    p = np.percentile([i[6][0] for i in position[:int(len(position))]], 50)

    e = d.datetime.now()

    print(len(position))
    print(len(np.unique(position)))

    print(m-s)
    print(e-m)

def chaining():
    # chained_positions = [[] for _ in range(len(np.unique([p for p in self.positions[6]])))]
    kk = [1,2,3,3,2,2,1,1,1,3,3,40,4,4,30,30,59,59,234,2,9,9,6,6,5,2,3,1,101,293,48,28,23,23]
    chained_positions = [[]]
    for p in kk:
        print(chained_positions)
        for pos in chained_positions:
            if pos == []:   #new
                pos.append(p)
                chained_positions.append([])
                break
            if p == pos[0]:  #matched
                pos.append(p)
                break
    return chained_positions[:-1]

def two_points(joint_a, joint_b):
    points = [[] for _ in range(3)]
    dense = 100

    for prop in range(dense+1):
        tmp_joint = [i - (i - j)* prop/dense for i, j in zip(joint_a, joint_b)]
        tmp_pos = fk_dh(tmp_joint)
        for i in range(3):
            points[i].append(tmp_pos[i])
        print(tmp_pos)
    return points

def draw(points):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax2 = Axes3D(fig)

    z = np.linspace(0,13,1000)
    x = 5*np.sin(z)
    y = 5*np.cos(z)
    zd = 13*np.random.random(100)
    xd = 5*np.sin(zd)
    yd = 5*np.cos(zd)
    ax2.scatter3D(points[0][1:-2], points[1][1:-2], points[2][1:-2], cmap='Blues')
    ax2.scatter3D(points[0][0], points[1][0], points[2][0], cmap='Reds')
    ax2.scatter3D(points[0][-1], points[1][-1], points[2][-1], cmap='Reds')
    # ax2.plot3D(x,y,z,'gray')    #繪製空間曲線
    plt.show()

def ikpy_test():
    from ikpy.chain import Chain
    import ikpy.utils.plot as plot_utils

    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], last_link_vector=[0, 0, 0])#, active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)

    target = [0.5545, 0.0, 0.6245]

    ini1 = [ 0.7,  0.8, -1.3, -1.5, -2.8,  0. ,  0. ]
    ini2 = [ 2.2, -0.2, -2.3, -1.5, -2.3,  3.5,  0. ]

    work_joints:list = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
    print([p[3] for p in chain.forward_kinematics([0, *work_joints, 0, 0])[:3]])

    result = chain.inverse_kinematics(target, [0, 0, -1], orientation_mode=None , initial_position=[0, *ini1, 0, 0])[1:8]
    # print(result)
    print([p[3] for p in chain.forward_kinematics([0, *result, 0, 0])[:3]])

    result = chain.inverse_kinematics(target, [0, 0, -1], orientation_mode=None , initial_position=[0, *ini2, 0, 0])[1:8]
    # print(result)
    print([p[3] for p in chain.forward_kinematics([0, *result, 0, 0])[:3]])


if __name__ == '__main__':
    #[ 0.5545 0  0.7315]
    joint_a:list = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
    pos_a = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]

    joint_b:list = [ 2.2,  0.3, -2.3, -2. , -2.8,  2.5,  0. ]
    pos_b = [ 0.5471, -0.0024,  0.6091]

    # pos_a = fk_dh(joint_a)
    # pos_b = fk_dh(joint_b)
    # print(pos_a)
    # print(pos_b)
    # pos_c = [(i+j)/2 for i, j in zip(pos_a, pos_b)]
    # pos_c = pos_b

    # joint_c = linear_interpolation(joint_a, joint_b, pos_a, pos_b, pos_c)
    # joint_c = approximation(joint_a, pos_c)

    # find_all_posture(joint_a, pos_a)

    # print(fk_dh(joint_a))
    # iktable = IKTable('table2')
    # print(iktable.positions[0])

    # draw(two_points(joint_a, joint_b))

    ikpy_test()
