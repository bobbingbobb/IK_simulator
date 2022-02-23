import os, time
import numpy as np
import random as r
import math as m
import datetime as d
from collections import namedtuple, defaultdict
from itertools import combinations
import copy as c


from sympy import Point3D, Plane, symbols, cos, sin
from sympy.matrices import Matrix, eye
from sympy import pprint

from ik_simulator import IKTable, IKSimulator
from utilities import *


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
    return np.array([p[3] for p in mat[:3]])

def fk_sympy(joints:list):
    # dh: joint(theta), distance between axes(-1 a), movement on axis(d), angle(-1 alpha)
    dh = np.array([[0.0,     0.0, 0.333,     0.0],\
                   [0.0,     0.0,   0.0, -m.pi/2],\
                   [0.0,     0.0, 0.316,  m.pi/2],\
                   [0.0,  0.0825,   0.0,  m.pi/2],\
                   [0.0, -0.0825, 0.384,  -m.pi/2],\
                   [0.0,     0.0,   0.0,  m.pi/2],\
                   [0.0,   0.088, 0.107,  m.pi/2]])

    # dh[:,0] = joints
    mat = eye(4)
    for i in range(7):
        dh_t = Matrix([[cos(joints[i])               , -sin(joints[i])               ,  0             ,  dh[i,1]               ],\
        		       [sin(joints[i])*cos(dh[i,3]),  cos(joints[i])*cos(dh[i,3]), -sin(dh[i,3]), -dh[i,2]*sin(dh[i,3])],\
        		       [sin(joints[i])*sin(dh[i,3]),  cos(joints[i])*sin(dh[i,3]),  cos(dh[i,3]),  dh[i,2]*cos(dh[i,3])],\
        		       [0                              ,  0                              ,  0             ,  1                     ]])
        mat = mat.multiply(dh_t)
        # print([p[3] for p in mat[:3]])
        # print(mat)

    pprint(mat)
    return mat

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

def spatial_interpolation(target):
    pass

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

def even_distribute(joints, dense=10, ind=0, agg=0):
    points=[]
    cp_joints = c.deepcopy(joints)
    # print(cp_joints)

    if ind+1 == len(joints):
        if not agg == 0:
            cp_joints[ind] = [j * (dense-agg)/dense for j in joints[ind]]
            tmp_joint = [np.sum(q) for q in np.array(cp_joints).T]
            tmp_pos = fk_dh(tmp_joint)
            # print(tmp_pos)
            points.append(tmp_pos)
    else:
        for prop in range(dense):
            if agg+prop > dense:
                break
            cp_joints[ind] = [j * prop/dense for j in joints[ind]]
            points += even_distribute(cp_joints, dense=dense, ind=(ind+1), agg=agg+prop)
            # print(points)
    return points

def two_points(joints, dense=100):
    points = []

    for prop in range(dense+1):
        tmp_joint = [i - (i - j)* prop/dense for i, j in zip(joints[0], joints[1])]
        tmp_pos = fk_dh(tmp_joint)
        points.append(tmp_pos)
        # print(tmp_pos)
    return points

def four_points(joints, dense=10):
    points = []

    for p1 in range(dense+1):
        for p2 in range(dense+1):
            if p1+p2 > dense:
                continue
            for p3 in range(dense+1):
                if p1+p2+p3 > dense:
                    continue
                p4 = dense - (p1 + p2 + p3)

                tmp_joint = [(q1*p1 + q2*p2 + q3*p3 + q4*p4)/dense for q1, q2, q3, q4 in [j for j in np.array(joints).T]]
                tmp_pos = fk_dh(tmp_joint)
                points.append(tmp_pos)
                # print(tmp_pos)
    return points

def draw(points, origin=None):
    # print(len(points))
    # print(origin[:-1])
    points = np.array(points).T
    origin = np.array(origin).T
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax2 = Axes3D(fig)


    ax2.scatter3D(points[0], points[1], points[2], c='grey')
    if not origin is None:
        # ax2.scatter3D(origin[0][:-2], origin[1][:-2], origin[2][:-2], c='blue')
        ax2.scatter3D(origin[0][:-3], origin[1][:-3], origin[2][:-3], c='blue')
        ax2.scatter3D(origin[0][-3], origin[1][-3], origin[2][-3], c='red')
        ax2.scatter3D(origin[0][-2], origin[1][-2], origin[2][-2], c='green')
        ax2.scatter3D(origin[0][-1], origin[1][-1], origin[2][-1], c='orange')

    # z = np.linspace(0,13,1000)
    # x = 5*np.sin(z)
    # y = 5*np.cos(z)
    # ax2.plot3D(x,y,z,'gray')    #繪製空間曲線
    plt.show()

def within(target, near_4_point):
    from sympy import Point3D, Plane

    # print(list(combinations([_ for _ in range(4)], 3)))

    for i in range(4):
        # print(i)
        plane_point = []
        for j in range(4):
            if i == j:
                outter_point = near_4_point[j]
            else:
                plane_point.append(near_4_point[j])

        plane = Plane(Point3D(plane_point[0]), Point3D(plane_point[1]), Point3D(plane_point[2]))

        dir = lambda x, y, z: eval(str(plane.equation()))

        tar_sgn = np.sign(dir(target[0], target[1], target[2]))
        out_sgn = np.sign(dir(outter_point[0], outter_point[1], outter_point[2]))

        if not tar_sgn == out_sgn:
            break
    else:
        return True

    print('not inside')
    return False

    # print(float(plane.equation(x=1.0e-6, y=1.0e-6, z=1.0e-6)))
    # equ = lambda x, y, z: eval(str(plane.equation()))
    # print(equ(0,0,0))

def int_approx(posture, target):
    target = np.array(target)

    jo = []
    di = []
    ori_diff = []
    time = []
    count = 0
    worst = 0
    num = 0
    total = 0
    for result in posture:
        check = True
        if len(result) > 1:

            joint = []
            diff = 1
            num += 1

            for ind in list(combinations(range(len(result)), 2)):
                origin = [result[i][0][6] for i in ind]

                vec = [origin[0]-target, origin[1]-target]
                # side = np.dot(vec[0]/np.linalg.norm(vec[0]), vec[1]/np.linalg.norm(vec[1]))
                # print(side)

                diff2 = min([np.linalg.norm(vec[0]), np.linalg.norm(vec[1])])
                print(diff2)
                # if side < -0.5:
                # print([result[i][1] for i in ind])

                s = d.datetime.now()

                # joint_int, diff_int = interpolate(result[ind[0]], result[ind[1]], target)
                # print('interpolate: ', diff_int/diff2, diff_int)

                joint_approx, diff_approx = approx_iter(result[ind[0]], result[ind[1]], target)
                print('       true: ', diff_approx/diff2, diff_approx)
                e = d.datetime.now()
                time.append(e-s)

                if diff_approx <= diff:
                # if diff_int <= diff:
                    diff = diff_approx
                    joint = joint_approx
                    # diff = diff_int
                    # joint = joint_int

                if diff/diff2 < 0.8:
                    # continue
                    if check:
                        count += 1
                        check = False

                total += 1
                if diff/diff2 > 1.03:
                    worst += 1

                p = even_distribute([result[i][1] for i in ind], dense=20)

                # origin.append(fk_dh(joint_int))
                origin.append(fk_dh(joint_approx))
                # origin.append(p[np.argmin([np.linalg.norm(pp - target) for pp in p])])
                origin.append(target)
                # draw(p, origin)

            di.append(diff)
            jo.append(joint)
            ori_diff.append(np.mean([np.linalg.norm(np.array(re[0][6])-target) for re in result]))


    # print(count, num)
    e = d.datetime.now()
    # print(e-s)

    message = {}
    # message['target'] = target
    message['posture'] = len(posture)
    message['work_p'] = num
    message['work_well'] = count
    message['worst%'] = worst/total if not total==0 else 0
    message['origin_diff'] = np.mean(ori_diff)
    message['mean_diff'] = np.mean(np.array(di))
    message['avg. time'] = np.mean(np.array(time))

    return message

def print_points(postures, target):

    for result in postures:
        length = 2
        if len(result) > length-1:
            # draw([i[0][6] for i in result], target)
            print([re[1] for re in result])
            for ind in list(combinations(range(len(result)), length)):
                print(ind)
                origin = [result[i][0][6].tolist() for i in ind]
                origin.append(target)
                # p = two_points([result[i][1] for i in ind], dense=10)
                # p = four_points([result[i][1] for i in ind], dense=10)
                p = even_distribute([result[i][1] for i in ind])
                # print(p)
                draw(p, origin)
                # break
            # break

def two_point_func(post_1, post_2, target):
    s = d.datetime.now()

    npp = np.linalg.norm(post_1[0][6]-post_2[0][6])
    n1t = np.linalg.norm(post_1[0][6]-target)
    n2t = np.linalg.norm(post_2[0][6]-target)

    if n1t < npp/2:
        tmp_joint = approx_iter(n1t/npp, post_1, post_2, target)
    elif n2t < npp/2:
        tmp_joint = approx_iter(n2t/npp, post_2, post_1, target)
    else:
        tmp_joint = approx_iter(0.5, post_1, post_2, target)

    print(d.datetime.now()-s)

    return tmp_joint

# def approx_iter(w1, post_1, post_2, target):
def approx_iter(post_1, post_2, target):
    # s = d.datetime.now()

    w1 = 0.5
    offset = 0.24
    diff = np.linalg.norm(np.array(post_1[0][6]) - target)
    tmp_joint = [q1*w1 + q2*(1-w1) for q1, q2 in zip(post_1[1], post_2[1])]

    while offset > 0.001:
        reverse = 0
        while reverse < 2:
            w1 += offset
            # print(offset)
            if w1 > 1 or w1 < 0:
                w1 -= offset
                print('out!')
                break
            pre_joint = tmp_joint
            tmp_joint = [q1*w1 + q2*(1-w1) for q1, q2 in zip(post_1[1], post_2[1])]
            pre_diff = diff
            diff = np.linalg.norm(fk_dh(tmp_joint) - target)
            if diff >= pre_diff:
                offset *= -1
                reverse += 1

            # print(diff)
            # draw(p, [fk_dh(tmp_joint)])
        else:
            w1 += offset
            tmp_joint = pre_joint
            diff = pre_diff

        offset *= abs(offset)

    # m = d.datetime.now()

    # e = d.datetime.now()
    # print(m-s, e-m)
    return tmp_joint, diff

def interpolate(post_1, post_2, target):
    s = d.datetime.now()

    full = np.array(post_2[0][6]) - np.array(post_1[0][6])
    part1 = target - np.array(post_1[0][6])

    w1 = np.dot(part1, full) / (np.linalg.norm(full) ** 2)
    tmp_joint = [q1*w1 + q2*(1-w1) for q1, q2 in zip(post_1[1], post_2[1])]
    # print(tmp_joint)

    diff = np.linalg.norm(fk_dh(tmp_joint) - target)

    print(d.datetime.now()-s)

    return tmp_joint, diff

def run_within(iter):
    ik_simulator = IKSimulator(algo='ikpy')
    num = []
    mes = defaultdict(list)

    for i in range(iter):
        x = round(r.uniform(-0.855, 0.855), 4)
        y = round(r.uniform(-0.855, 0.855), 4)
        z = round(r.uniform(-0.36, 1.19), 4)
        target = [x, y, z]
        result = ik_simulator.find(target)
        if result:
            message = int_approx(result, target)
            print(message)
            for k,v in message.items():
                mes[k].append(v)
    result = {}
    for k, v in mes.items():
        if k == 'avg. time':
            result[k] = np.mean(v)
        else:
            result[k] = np.nanmean(v)

    messenger(result)

def ikpy_test():
    from ikpy.chain import Chain
    import ikpy.utils.plot as plot_utils

    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], last_link_vector=[0, 0, 0])#, active_links_mask=[False, True, True, True, True, True, True, True, False, False])
    # print(chain)

    target = [0.5545, 0.0, 0.6245]

    ini1 = [-0.3,  0.8,  0.7, -0.5,  0.2,  0. ,  0. ]
    ini2 = [ 2.2,  0.3, -2.3, -2. ,  2.2,  2.5,  0. ]
    inifar1 = [ -2.2,  0.0, 0.0, -0.3 ,  1.5,  0.2,  0. ]

    work_joints = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
    print([p[3] for p in chain.forward_kinematics([0, *work_joints, 0, 0])[:3]])

    s = d.datetime.now()
    # result = chain.inverse_kinematics(target, initial_position=[0]*10)[1:8]
    result = chain.inverse_kinematics(target, initial_position=[0, *inifar1, 0, 0])[1:8]

    e = d.datetime.now()
    print(e-s)

    result = chain.inverse_kinematics(target, initial_position=[0, *ini1, 0, 0])[1:8]
    # print(result)
    print([p[3] for p in chain.forward_kinematics([0, *result, 0, 0])[:3]])

    s = d.datetime.now()
    print(s-e)

    result = chain.inverse_kinematics(target, initial_position=[0, *ini2, 0, 0])[1:8]
    # print(result)
    print([p[3] for p in chain.forward_kinematics([0, *result, 0, 0])[:3]])

    e = d.datetime.now()
    print(e-s)

def dense_test(target):
    iktable = IKTable('dense')
    # result = iktable.query(target)
    ik_simulator = IKSimulator(algo='ikpy')
    # result = ik_simulator.find(target)

    for _ in range(10):
        x = r.uniform(0.2, 0.25)
        y = r.uniform(0.45, 0.5)
        z = r.uniform(0.3, 0.35)

        target = [x, y, z]

        print(len(iktable.query(target)), len(ik_simulator.find(target)))
        print()

if __name__ == '__main__':
    #[ 0.5545 0  0.7315]
    target = [0.5545, 0.0, 0.6245]
    joint_a:list = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
    pos_a = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]

    joint_b:list = [ 2.2,  0.3, -2.3, -2. , -2.8,  2.5,  0. ]
    pos_b = [ 0.5471, -0.0024,  0.6091]



    # ik_simulator = IKSimulator(algo='ikpy')
    # x = round(r.uniform(-0.855, 0.855), 4)
    # y = round(r.uniform(-0.855, 0.855), 4)
    # z = round(r.uniform(-0.36, 1.19), 4)
    # target = [x, y, z]
    # target = pos_a
    # result = ik_simulator.find(target)
    # if result:
    #     # print_points(result, target)
    #     int_approx(result, target)

    # run_within(500)

    target = [0.22, 0.47, 0.32]
    dense_test(target)
