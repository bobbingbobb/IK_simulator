import os
import numpy as np
import datetime as d
import random as r
import copy as c

from constants import *
from utilities import *
from data_gen import Robot, DataCollection
from ik_simulator import IKTable, IKSimulator

from ikpy.chain import Chain
import ikpy.utils.plot as plot_utils

chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], last_link_vector=[0, 0, 0])#, active_links_mask=[False, True, True, True, True, True, True, True, False, False])


def fully_covered(iter):
    # iktable = IKTable('rtree_20')
    # iktable = IKTable('dense')
    iktable = IKTable('full_jointonly_0')
    for t in iktable.table:
        print(t.bounds)

    avg = []
    n_avg = []
    dev = []
    pos = []
    for i in range(iter):
        counter = 0
        n_counter = 0
        for _ in range(1000):

            # x = round(r.uniform(-0.855, 0.855), 4)
            # y = round(r.uniform(-0.855, 0.855), 4)
            # z = round(r.uniform(-0.36, 1.19), 4)
            # x = round(r.uniform(0.2, 0.25), 4)
            # y = round(r.uniform(0.45, 0.5), 4)
            # z = round(r.uniform(0.3, 0.35), 4)
            x = round(r.uniform(0.2, 0.21), 4)
            y = round(r.uniform(0.4, 0.41), 4)
            z = round(r.uniform(0.3, 0.31), 4)

            result = iktable.dot_query([x, y, z])
            if len(result):#in range
                pos.append(len(result))
                counter += 1
                # print(result)
            else:
                n_counter += 1
                dev.append(np.linalg.norm(np.array(list(iktable.table[0].nearest([x, y, z], objects=True))[0].bounds[::2]) - np.array([x, y, z])))
            # break
        # break
        print(i+1, counter)
        avg.append(counter)
        n_avg.append(n_counter)

    mes = {}
    mes['pos'] = np.mean(pos)
    mes['avg'] = np.mean(avg)
    mes['n_avg'] = np.mean(n_avg)
    mes['len'] = np.mean(dev)
    mes['worst'] = max(dev) if dev else 0
    messenger(mes)

def current_ik_speed(iter):
    time = []
    dev = []
    n = 0
    robot = Robot()
    q = np.zeros(7)
    for j in range(6):
        q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)

    for i in range(iter):
        x = round(r.uniform(-0.855, 0.855), 4)
        y = round(r.uniform(-0.855, 0.855), 4)
        z = round(r.uniform(-0.36, 1.19), 4)
        target = [x, y, z]

        s = d.datetime.now()
        tmp_joint = chain.inverse_kinematics(target, initial_position=[0]*10)[1:8]
        # tmp_joint = chain.inverse_kinematics(target, initial_position=[0, *q, 0, 0])[1:8]
        e = d.datetime.now()

        fk, _ = robot.fk_dh(tmp_joint)
        deviation = np.linalg.norm(np.array(target) - np.array([round(p, 4) for p in fk]))
        print(target, fk)
        print(deviation)
        if not target == [round(p, 4) for p in fk]:
            dev.append(deviation)
            n += 1
        time.append(e-s)

    mes = {}
    mes['deviation'] = np.mean(dev)
    mes['failed'] = n
    mes['worst'] = max(dev)
    mes['avg. time'] = np.mean(np.array(time))

    print(mes)
    print(np.mean(np.array(time)))

def draw(dataset, name):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax2 = Axes3D(fig)



    # data = np.load(RESULT_FOLDER+dataset+'/'+name+'.npy', allow_pickle=True)
    data = np.load(RESULT_FOLDER+'posture_num_1.npy', allow_pickle=True)[::4]
    # print(data.shape)
    pos = np.array([p[0] for p in data]).T
    color = [p[1] for p in data]
    # color = [p[1] if p[1]<300 else 300 for p in data]

    # color = max(color)
    # print(pos)

    jo = np.array([[ 0.00000000e+00,  0.00000000e+00,  1.40000000e-01],
                   [ 0.00000000e+00,  0.00000000e+00,  3.33000000e-01],
                   [ 0.00000000e+00, -1.18178416e-17,  5.26000000e-01],
                   [ 8.25000000e-02, -1.18178416e-17,  6.49000000e-01],
                   [ 2.07000000e-01, -6.76617357e-18,  7.31500000e-01],
                   [ 4.66500000e-01, -2.26559658e-17,  7.31500000e-01],
                   [ 5.54500000e-01, -2.92078262e-17,  6.24500000e-01]])
    jo = jo.T

    plt.colorbar(ax2.scatter3D(pos[0], pos[1], pos[2], cmap='Blues', c=color))
    ax2.scatter3D(jo[0], jo[1], jo[2], c='red')
    # deviation = [p[0] for d in data for p in d]
    # print(len(deviation))
    # print(deviation[0])
    # print(deviation[-1])

    # plt.figure(1)
    # plt.scatter('deviation', 'num', data = deviation)
    plt.show()

def posture_num(iter):
    start = d.datetime.now()
    from ikpy.chain import Chain
    import ikpy.utils.plot as plot_utils
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], last_link_vector=[0, 0, 0], active_links_mask=[False, True, True, True, True, True, True, True, False, False])
    robot = Robot()

    thres = 0.5
    q = np.zeros(7)
    target = [0.0, 0.0, 0.0]
    posture = []

    for x in range(-855, 855, 100):
        target[0] = x/1000
        for y in range(-855, 855, 100):
            target[1] = y/1000
            for z in range(-360, 1190, 100):
                target[2] = z/1000
                p_list = []
                for i in range(iter):
                    print(i+1, target)
                    post = []
                    end = 0
                    pout = 0
                    while end < 100 and pout < 10:

                        for j in range(6):
                            q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)

                        try:
                            joint = chain.inverse_kinematics(target, initial_position=[0, *q, 0, 0])[1:8]
                            if np.linalg.norm(robot.fk_dh(joint)[0] - target) > 0.05:
                                print(target, len(post))
                                pout += 1
                                continue

                            for type in post:
                                for j_joint, j_type in zip(joint, type):
                                    if abs(j_joint-j_type) >= thres:
                                        break
                                else:
                                    break
                            else:
                                post.append(joint)
                                end = 0

                            end += 1

                        except ValueError:
                            print('Error raised')
                            continue

                    p_list.append(len(post))
                num = np.mean(p_list)
                print(num)
                if num:
                    posture.append([c.copy(target), num])

        np.save(RESULT_FOLDER+'posture_num_'+str(iter), posture)
    end = d.datetime.now()
    print('done. duration: ', end-start)

def query_time():
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)
    dataset = 'rtree_20'
    ik_simulator = IKSimulator(algo='ikpy', dataset=dataset)
    iktable = IKTable(dataset)


    if dataset == 'rtree_20':
        res = [-0.855, 0.855, -0.855, 0.855, -0.36, 1.19]
    elif dataset == 'dense':
        res = [0.2, 0.25, 0.45, 0.5, 0.3, 0.35]
    elif dataset == 'full_jointonly_fixed1':
        res = [0.2, 0.215, 0.4, 0.415, 0.3, 0.315]
    # filename = RESULT_FOLDER+dataset+'/'+name

    x = round(r.uniform(res[0], res[1]), 4)
    y = round(r.uniform(res[2], res[3]), 4)
    z = round(r.uniform(res[4], res[5]), 4)
    target = [x, y, z]


    s = d.datetime.now()
    joint = iktable.query(target)[0][1]
    result = ik_simulator.ikpy_run(joint, target)
    e = d.datetime.now()
    query = e - s
    print(np.linalg.norm(ik_simulator.fk(joint)-np.array(target)))

    s = d.datetime.now()
    joint = ik_simulator.find(target)[0][0][0][1] #classify
    result = ik_simulator.ikpy_run(joint, target)
    e = d.datetime.now()
    classify = e - s
    print(np.linalg.norm(ik_simulator.fk(joint)-np.array(target)))

    s = d.datetime.now()
    joint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = chain.inverse_kinematics(target)[1:8]
    e = d.datetime.now()
    ikpy = e - s
    print(np.linalg.norm(ik_simulator.fk(joint)-np.array(target)))

    print(query)
    print(classify)
    print(ikpy)

if __name__ == '__main__':
    print('start')
    start = d.datetime.now()

    # fully_covered(1)
    # current_ik_speed(1000)
    # posture_num(1)
    # draw('rtree_20', 'inter_300_post')
    query_time()

    print('duration: ', d.datetime.now()-start)
    print('end')
