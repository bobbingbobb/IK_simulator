import os, time
import numpy as np
import random as r
import math as m
import datetime as d
from collections import namedtuple, defaultdict
from itertools import combinations
import copy as c


from ik_simulator import IKTable, IKSimulator
from data_gen import Robot
from utilities import *


ik_dict = {}

robot = Robot()

def linear_interpolation(joint_a, joint_b, pos_a, pos_b, pos_c):
    dist_a = np.linalg.norm(pos_a - pos_c)
    dist_b = np.linalg.norm(pos_b - pos_c)

    prop_a = dist_a/(dist_a+dist_b)
    print(prop_a)

    tmp_joint = [i - (i - j)* prop_a for i, j in zip(joint_a, joint_b)]
    tmp_pos = robot.fk_dh(tmp_joint)[0]
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
            tmp_pos = robot.fk_dh(tmp_joint)[0]
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
        # if len(result) > 5 and len(result) < 10:
        if len(result) == 3:

            joint = []
            diff = 1
            num += 1

            for ind in list(combinations(range(len(result)), 2)):
                origin = [result[i][0][6] for i in ind]

                vec = [origin[0]-target, origin[1]-target]
                # side = np.dot(vec[0]/np.linalg.norm(vec[0]), vec[1]/np.linalg.norm(vec[1]))
                # print(side)

                diff2 = min([np.linalg.norm(vec[0]), np.linalg.norm(vec[1])])
                # print(diff2)
                # if side < -0.5:
                # print([result[i][1] for i in ind])

                s = d.datetime.now()

                joint_int, diff_int = interpolate(result[ind[0]], result[ind[1]], target)
                # print('interpolate: ', diff_int/diff2, diff_int)
                if diff_int <= diff:
                    diff = diff_int
                    joint = joint_int

                # joint_approx, diff_approx = approx_iter(result[ind[0]], result[ind[1]], target)
                # print('       true: ', diff_approx/diff2, diff_approx)
                # if diff_approx <= diff:
                    # diff = diff_approx
                    # joint = joint_approx

                e = d.datetime.now()
                time.append(e-s)

                if diff/diff2 < 0.8:
                    # continue
                    if check:
                        count += 1
                        check = False

                total += 1
                if diff/diff2 > 1.03:
                    worst += 1

                # p = even_distribute([result[i][1] for i in ind], dense=20)

                # origin.append(robot.fk_dh(joint_int)[0])
                # origin.append(robot.fk_dh(joint_approx)[0])
                # origin.append(p[np.argmin([np.linalg.norm(pp - target) for pp in p])])
                # origin.append(target)
                # draw(p, origin)

            di.append(diff)
            jo.append(joint)
            # ori_diff.append(np.mean([np.linalg.norm(np.array(re[0][6])-target) for re in result]))
            ori_diff.append(np.mean([np.linalg.norm(np.array(re[0][6])-target) for re in result]))
            # break

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
            diff = np.linalg.norm(robot.fk_dh(tmp_joint) - target)
            if diff >= pre_diff:
                offset *= -1
                reverse += 1

            # print(diff)
            # draw(p, [robot.fk_dh(tmp_joint)[0]])
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
    # s = d.datetime.now()

    full = np.array(post_2[0][6]) - np.array(post_1[0][6])
    part1 = target - np.array(post_1[0][6])

    w1 = np.dot(part1, full) / (np.linalg.norm(full) ** 2)
    tmp_joint = [q1*w1 + q2*(1-w1) for q1, q2 in zip(post_1[1], post_2[1])]
    # print(tmp_joint)

    diff = np.linalg.norm(robot.fk_dh(tmp_joint)[0] - target)

    # print(d.datetime.now()-s)

    return tmp_joint, diff

def run_within(iter):
    ik_simulator = IKSimulator(algo='ikpy')
    num = []
    mes = defaultdict(list)

    for i in range(iter):
        # x = round(r.uniform(-0.855, 0.855), 4)
        # y = round(r.uniform(-0.855, 0.855), 4)
        # z = round(r.uniform(-0.36, 1.19), 4)

        x = r.uniform(0.2, 0.25)
        y = r.uniform(0.45, 0.5)
        z = r.uniform(0.3, 0.35)
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
    from ikpy.link import DHLink as Link
    import ikpy.utils.plot as plot_utils

    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)
    j = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    result = chain.inverse_kinematics([0.5, 0.0, 0.6], initial_position=[0, *j, 0])[1:8]
    # result = chain.inverse_kinematics([0.2, 0.4, 0.3])[1:8]
    print(result)
    print([p[3] for p in chain.forward_kinematics([0, *result, 0])[:3]])


    # work_joints = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
    # print([p[3] for p in chain.forward_kinematics([0, *work_joints, 0])[:3]])

    robot = Robot()
    print(robot.fk_dh(result)[0])
    # dhlink = Link(robot.dh)

    # target = [0.5545, 0.0, 0.6245]
    #
    # ini1 = [-0.3,  0.8,  0.7, -0.5,  0.2,  0. ,  0. ]
    # ini2 = [ 2.2,  0.3, -2.3, -2. ,  2.2,  2.5,  0. ]
    # inifar1 = [ -2.2,  0.0, 0.0, -0.3 ,  1.5,  0.2,  0. ]
    #
    # work_joints = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
    # print([p[3] for p in chain.forward_kinematics([0, *work_joints, 0, 0])[:3]])
    #
    # s = d.datetime.now()
    # result = chain.inverse_kinematics(target, initial_position=[0]*10)[1:8]
    # # result = chain.inverse_kinematics(target, initial_position=[0, *inifar1, 0, 0])[1:8]
    #
    # e = d.datetime.now()
    # print(e-s)
    #
    # result = chain.inverse_kinematics(target, initial_position=[0, *ini1, 0, 0])[1:8]
    # # print(result)
    # print([p[3] for p in chain.forward_kinematics([0, *result, 0, 0])[:3]])
    #
    # s = d.datetime.now()
    # print(s-e)
    #
    # result = chain.inverse_kinematics(target, initial_position=[0, *ini2, 0, 0])[1:8]
    # # print(result)
    # print([p[3] for p in chain.forward_kinematics([0, *result, 0, 0])[:3]])
    #
    # e = d.datetime.now()
    # print(e-s)

def ikpy_draw():
    from ikpy.chain import Chain
    from ikpy.link import DHLink as Link

    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)

    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10,10))
    # fig.set_size_inches(10, 10, True)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.855, 0.855])
    ax.set_ylim([-0.855, 0.855])
    ax.set_zlim([-0.36, 1.19])

    def init():
        # label = ax.text(.5, .5, '', fontsize=15)
        rob = chain.plot([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], ax)
        return rob

    def update(i):
        print('upup')
        t = [0.2092, -0.8056, 0.1131]
        ax.clear()
        ax.set_xlim([-0.855, 0.855])
        ax.set_ylim([-0.855, 0.855])
        ax.set_zlim([-0.36, 1.19])
        ax.scatter3D(t[0], t[1], t[2], c='red')
        diff = np.linalg.norm([p[3] for p in chain.forward_kinematics([0.0, *joint_list[i], 0.0])[:3]]-np.array(t))
        ax.text(0.5,0.5,2.1, str(i), fontsize=15)
        ax.text(0.5,0.5,2, str(diff), fontsize=15)
        ax.set_ylabel(str(t), fontsize=15)
        return chain.plot([0.0+i*0.05]*9, ax)


    name = 'move'
    k = 0
    filename = name+str(k)+'.gif'
    while os.path.exists(filename):
        k += 1
        filename = name+str(k)+'.gif'


    ani = FuncAnimation(fig, update, frames = 10, interval = 200, init_func=init, blit=False)
    # ani.save(filename, writer='imagemagick', fps=0.5)
    plt.show()

def dense_test(target, iter):
    iktable = IKTable('dense')
    # result = iktable.query(target)
    ik_simulator = IKSimulator(algo='ikpy')
    # result = ik_simulator.find(target)

    for _ in range(iter):
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

    # ikpy_test()
    ikpy_draw()

    # ik_simulator = IKSimulator(algo='ikpy')
    # x = round(r.uniform(-0.855, 0.855), 4)
    # y = round(r.uniform(-0.855, 0.855), 4)
    # z = round(r.uniform(-0.36, 1.19), 4)
    # target = [x, y, z]
    # target = pos_a
    target = [0.22, 0.47, 0.32]
    # result = ik_simulator.find(target)
    # print(len(result))
    # for r in result:
    #     if len(r) > 10:
    #         print(len(r))
    # if result:
    #     print_points(result, target)
        # int_approx(result, target)

    # run_within(1)

    # dense_test(target, 1)
