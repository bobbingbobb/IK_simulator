import os
import numpy as np
import datetime as d
import random as r
import copy as c
from rtree import index


from constants import *
from utilities import *
from data_gen import Robot, DataCollection
from ik_simulator import IKTable, IKSimulator

import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

from ikpy.chain import Chain
import ikpy.utils.plot as plot_utils



def fully_covered(iter, dataset):
    # iktable = IKTable('rtree_20')
    # iktable = IKTable('dense')
    # iktable = IKTable('full_jointonly_0')
    # for t in iktable.table:
    #     print(t.bounds)
    p = index.Property(dimension=3)
    idx = index.Index(os.path.join(RAW_DATA_FOLDER, dataset), properties=p)
    res = idx.bounds
    print(res)

    avg = []
    n_avg = []
    dev = []
    pos = []
    time = []
    for i in range(iter):
        counter = 0
        n_counter = 0
        for _ in range(1000):

            x = round(r.uniform(res[0], res[3]), 4)
            y = round(r.uniform(res[1], res[4]), 4)
            z = round(r.uniform(res[2], res[5]), 4)
            target = [x, y, z]

            qs = d.datetime.now()
            # result = iktable.dot_query([x, y, z])
            qu_res = idx.intersection(c.copy(target), objects=True)
            qe = d.datetime.now()
            time.append(qe-qs)
            result = [item.object for item in qu_res]
            if len(result):#in range
                pos.append(len(result))
                counter += 1
                # print(result)
            else:
                n_counter += 1
                dev.append(np.linalg.norm(np.array(list(idx.nearest(c.copy(target), objects=True))[0].bounds[::2]) - np.array([x, y, z])))
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
    mes['time'] = np.mean(time)
    messenger(mes)

def ik_iteration(iter):
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    robot = Robot()
    iteration = []
    deviation = []
    dev = [0.05, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0001, 0.0]
    origin = robot.fk_dh([0]*7)[0]

    for _ in range(iter):
        stat = 1
        while stat:
            x = round(r.uniform(-0.855, 0.855), 4)
            y = round(r.uniform(-0.855, 0.855), 4)
            z = round(r.uniform(-0.36, 1.19), 4)
            target = [x, y, z]
            diff = np.linalg.norm(np.array(origin) - np.array(target))
            # if diff < 0.05:
            #     continue
            tmp_joint, num, time, stat, joint_list = chain.inverse_kinematics(target)
        # print(len(joint_list))
        deviation.append(diff)
        i = 0
        it = []
        for dis in dev:
            tmp = 0
            while i < len(joint_list) and np.linalg.norm(robot.fk_dh(joint_list[i])[0] - np.array(target)) > dis:
                tmp += 1
                i += 1
            it.append(tmp)
        iteration.append(it)

    # np.save(RESULT_FOLDER+str(iter)+'iteration_dis', [dev, iteration])

    iteration = np.array(iteration).T
    message = {}
    message['diff'] = np.mean(deviation)
    for i in range(len(dev)):
        message[str(dev[i])] = np.mean(iteration[i])
    messenger(message)
    np.save(RESULT_FOLDER+str(iter)+'iteration_dis', message)


def current_ik_speed(iter):
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    time = []
    dev = []
    n = 0
    robot = Robot()
    q = np.zeros(7)
    for j in range(6):
        q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)

    for dis in range(50):
        dis /= 5000
        print(dis)
        for _ in range(iter):
            stat = 1
            while stat:
                x = round(r.uniform(-0.855, 0.855), 4)
                y = round(r.uniform(-0.855, 0.855), 4)
                z = round(r.uniform(-0.36, 1.19), 4)
                otarget = [x, y, z]
                joint, _, _, stat = chain.inverse_kinematics(otarget)
            # print(pos_alignment(chain.forward_kinematics(tmp_joint)[:3,3].tolist()), target)
            # print(pos_alignment(robot.fk_dh(tmp_joint[1:8])[0].tolist()), target)

            rp = np.array([r.uniform(-1, 1) for _ in range(3)])
            rp *= (dis/np.linalg.norm(rp))
            target = otarget + rp
            tmp_joint, ni, ntime, nstat = chain.inverse_kinematics(target, initial_position=joint)

            if not nstat:
                dev.append(dis)
                time.append(ni)
    np.save(RESULT_FOLDER+'distribution_range_iter_4', [time, dev])

    #     s = d.datetime.now()
    #     tmp_joint, ni, ntime, nstat = chain.inverse_kinematics(target, initial_position=[0]*10)[1:8]
    #     # tmp_joint = chain.inverse_kinematics(target, initial_position=[0, *q, 0, 0])[1:8]
    #     e = d.datetime.now()
    #
    #     fk, _ = robot.fk_dh(tmp_joint)
    #     deviation = np.linalg.norm(np.array(target) - np.array([round(p, 4) for p in fk]))
    #     print(target, fk)
    #     print(deviation)
    #     if not target == [round(p, 4) for p in fk]:
    #         dev.append(deviation)
    #         n += 1
    #     time.append(e-s)
    #
    # mes = {}
    # mes['deviation'] = np.mean(dev)
    # mes['failed'] = n
    # mes['worst'] = max(dev)
    # mes['avg. time'] = np.mean(np.array(time))
    #
    # print(mes)
    # print(np.mean(np.array(time)))

def draw_line():
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    data = np.load(RESULT_FOLDER+'distribution_range_iter_4.npy', allow_pickle=True)
    print(len(data[1]))
    i = 0
    iter = []
    dev = []

    for dis in range(50):
        dis /= 5000
        tmp = []
        while round(data[1][i],4) == dis:
            tmp.append(data[0][i])
            i += 1
            if i == len(data[0]):
                break
        dev.append(dis)
        iter.append(np.mean(tmp))
        print(len(tmp))

    # ra = 0.0
    # i_dis = np.argsort(data[1])
    # while i < len(i_dis):
    #     tmp_i = []
    #     tmp_d = []
    #     ra += 0.005
    #     while data[1][i_dis[i]] < ra:
    #         tmp_i.append(data[0][i_dis[i]])
    #         tmp_d.append(data[1][i_dis[i]])
    #         i += 1
    #         if i == len(i_dis):
    #             break
    #     iter.append(np.mean(tmp_i))
    #     dev.append(np.mean(tmp_d))

    # print(iter)
    # print(dev)
    #
    plt.scatter(iter, dev)
    # z = np.polyfit(iter, dev, 1)
    # p = np.poly1d(z)
    # plt.plot(iter,p(iter),"r--")
    plt.xlabel("iter (num)")
    plt.ylabel("dev (m)")
    plt.title('distance vs iteration')
    plt.savefig(RESULT_FOLDER+'dist_iter_4.png')
    plt.show()

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

def query_time(dataset, iter, threshold):
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)
    ik_simulator = IKSimulator(algo='ikpy', dataset=dataset)
    p = index.Property(dimension=3)
    idx = index.Index(os.path.join(RAW_DATA_FOLDER, dataset), properties=p)

    if dataset.startswith('rtree'):
        res = [-0.855, -0.855, -0.36, 0.855, 0.855, 1.19]
        if dataset.startswith('rtree_20'):
            dsf = 'rtree_20/'
        else:
            dsf = 'rtree_30/'
    elif dataset.startswith('dense'):
        res = [0.2, 0.45, 0.3, 0.25, 0.5, 0.35]
        dsf = 'dense/'
    elif dataset.startswith('full'):
        res = [0.2, 0.4, 0.3, 0.215, 0.415, 0.315]
        dsf = 'full/'

    filename = RESULT_FOLDER+dsf+'1compare_'+str(iter)+'_'+str(threshold).replace('.','')
    print(dataset+'_'+str(iter)+'_'+str(threshold))

    time_q = []

    time_n = []
    oridiff_n = []
    num_n = []

    time_c = []
    oridiff_c = []
    num_c = []

    time_i = []
    oridiff_i = []
    num_i = []

    for _ in range(iter):
        x = round(r.uniform(res[0], res[3]), 4)
        y = round(r.uniform(res[1], res[4]), 4)
        z = round(r.uniform(res[2], res[5]), 4)
        target = [x, y, z]
        # target = [0.2731, 0.175, -0.2938]
        # print(target)

        qs = d.datetime.now()
        # joint = ik_simulator.iktable.query(target)[0][1]
        qu_res = idx.nearest(c.copy(target), 1, objects=True)
        qe = d.datetime.now()
        query = qe - qs
        if query.seconds > 0.1:
            print(target, query)
        joint = [item.object for item in qu_res][0][1]
        result, nearby, ne_stat, nej_list = chain.inverse_kinematics(target, initial_position=[0, *joint, 0])
        ne_oridiff = np.linalg.norm(ik_simulator.fk(joint)-np.array(target))
        ne_diff = np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target))
        ne_n = len(nej_list)

        # # s = d.datetime.now()
        # # joint = ik_simulator.find(target)[0][0][0][1] #classify
        # result, classify, c_stat, cj_list = chain.inverse_kinematics(target, initial_position=[0, *joint, 0])
        # # e = d.datetime.now()
        # # classify = e - s
        # c_oridiff = np.linalg.norm(ik_simulator.fk(joint)-np.array(target))
        # c_diff = np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target))
        # c_n = len(cj_list)

        # # s = d.datetime.now()
        # joint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # result, ikpy, i_stat, ij_list = chain.inverse_kinematics(target, initial_position=[0, *joint, 0])
        # # e = d.datetime.now()
        # # ikpy = e - s
        # i_oridiff = np.linalg.norm(ik_simulator.fk(joint)-np.array(target))
        # i_diff = np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target))
        # i_n = len(ij_list)

        # print(nearby, ne_n, ne_diff)
        # print(classify, c_n, c_diff)
        # print(ikpy, i_n, i_diff)
        # print()

        # if c_n < i_n:
        #     print(target)
        #     print(c_diff, m_diff)

        if not ne_stat:
            time_n.append(nearby)
            oridiff_n.append(ne_oridiff)
            num_n.append(ne_n)

        # if not c_stat:
        #     time_c.append(classify)
        #     oridiff_c.append(c_oridiff)
        #     num_c.append(c_n)

        # if not i_stat:
        #     time_i.append(ikpy)
        #     oridiff_i.append(i_oridiff)
        #     num_i.append(i_n)

        time_q.append(query)

        # if ne_diff < threshold and i_diff < threshold:
    #     time_n.append(nearby)
    #     oridiff_n.append(ne_oridiff)
    #     num_n.append(ne_n)
    #     time_i.append(ikpy)
    #     oridiff_i.append(i_oridiff)
    #     num_i.append(i_n)
    #
    # filename = filename+'_all'

    message = {}
    message['nearby'] = len(time_n)
    # message['classify'] = len(time_c)
    # message['ikpy'] = len(time_i)
    message['query'] = np.mean(time_q)
    message['time_n'] = np.mean(time_n)
    # message['time_c'] = np.mean(time_c)
    # message['time_i'] = np.mean(time_i)
    message['oridiff_n'] = np.mean(oridiff_n)
    # message['oridiff_c'] = np.mean(oridiff_c)
    # message['oridiff_i'] = np.mean(oridiff_i)
    message['num_n'] = np.mean(num_n)
    # message['num_c'] = np.mean(num_c)
    # message['num_i'] = np.mean(num_i)

    np.save(filename, message)
    messenger(message)

def secondary_compare(dataset, iter, threshold):
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)
    robot = Robot()
    ik_simulator = IKSimulator(algo='ikpy', dataset=dataset)
    p = index.Property(dimension=3)
    idx = index.Index(os.path.join(RAW_DATA_FOLDER, dataset), properties=p)

    if dataset.startswith('rtree'):
        res = [-0.855, -0.855, -0.36, 0.855, 0.855, 1.19]
        if dataset.startswith('rtree_20'):
            dsf = 'rtree_20/'
        else:
            dsf = 'rtree_30/'
    elif dataset.startswith('dense'):
        res = [0.2, 0.45, 0.3, 0.25, 0.5, 0.35]
        dsf = 'dense/'
    elif dataset.startswith('full'):
        res = [0.2, 0.4, 0.3, 0.215, 0.415, 0.315]
        dsf = 'full/'

    filename = RESULT_FOLDER+dsf+'secondary_'+str(iter)+'_'+str(threshold).replace('.','')
    print(dataset+'_'+str(iter)+'_'+str(threshold))

    time_q = []
    time_c = []
    time_s = []
    post_num = []

    time_n = []
    oridiff_n = []
    num_n = []
    ee_dev = []
    ori_dev = []

    time_i = []
    oridiff_i = []
    num_i = []

    for _ in range(iter):
        # x = round(r.uniform(res[0], res[3]), 4)
        # y = round(r.uniform(res[1], res[4]), 4)
        # z = round(r.uniform(res[2], res[5]), 4)
        # target_pos = [x, y, z]
        #
        # target_ori = np.array([np.random() for _ in range(3)])
        # target_ori /= np.linalg.norm(target_ori)
        q = np.zeros(7)
        for j in range(6):
            q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)
        target_pos, target_ori = robot.fk_dh(q)

        s = d.datetime.now()
        # pos_info = [item.object for item in idx.intersection([t+offset for offset in (-0.025, 0.025) for t in target_pos], objects=True)]
        # if len(pos_info) < 20:
        #     pos_info = [item.object for item in idx.nearest(target_pos.tolist(), 50, objects=True)] if len(pos_info) < 20 else pos_info

        pos_info = ik_simulator.iktable.rtree_query(target_pos.tolist())

        # pos_info = [item.object for item in idx.nearest(target_pos.tolist(), 50, objects=True)]
        e = d.datetime.now()
        query = e - s

        s = d.datetime.now()
        nearby_postures = [pos_info[inds[0]] for inds in ik_simulator.posture_comparison_all_joint_sorted(pos_info)]#index
        e = d.datetime.now()
        classify = e - s

        post_num.append(len(nearby_postures))
        # print(len(nearby_postures))

        s = d.datetime.now()
        ori_tmp = 0
        for i, post in enumerate(nearby_postures):
            likeliness = np.dot(post[2], target_ori)
            if likeliness > ori_tmp:
            # if likeliness > 0.95:
                ori_tmp = likeliness
                joint = post[1]
                # post_num.append(i)
                # break
        e = d.datetime.now()
        outter_task = e - s

        result, nearby, ne_stat, nej_list = chain.inverse_kinematics(target_pos, initial_position=[0, *joint, 0])
        # result, ne_n, nearby = chain.inverse_kinematics(target_pos, target_ori, orientation_mode='X', initial_position=[0, *joint, 0])
        ne_oridiff = np.linalg.norm(ik_simulator.fk(joint)-np.array(target_pos))
        ne_pos, ne_ori = robot.fk_dh(result[1:8])
        ne_diff = np.linalg.norm(ne_pos-np.array(target_pos))
        ne_n = len(nej_list)

        joint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result, ikpy, i_stat, ij_list = chain.inverse_kinematics(target_pos, target_ori, orientation_mode='all', initial_position=[0, *joint, 0])
        i_oridiff = np.linalg.norm(ik_simulator.fk(joint)-np.array(target_pos))
        i_diff = np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target_pos))
        i_n = len(ij_list)

        if not ne_stat:
            time_q.append(query)
            time_c.append(classify)
            time_s.append(outter_task)
            time_n.append(nearby)
            oridiff_n.append(ne_oridiff)
            num_n.append(ne_n)
            ori_dev.append(ori_tmp)
            ee_dev.append(np.dot(target_ori, ne_ori))

        if not i_stat:
            time_i.append(ikpy)
            oridiff_i.append(i_oridiff)
            num_i.append(i_n)


    message = {}
    message['nearby'] = len(time_n)
    message['ikpy'] = len(time_i)
    message['post_num'] = np.mean(post_num)
    message['likeliness'] = np.mean(ee_dev)
    message['olikeliness'] = np.mean(ori_dev)
    message['time_q'] = np.mean(time_q)
    message['time_c'] = np.mean(time_c)
    message['time_s'] = np.mean(time_s)
    message['time_n'] = np.mean(time_n)
    message['time_i'] = np.mean(time_i)
    message['oridiff_n'] = np.mean(oridiff_n)
    message['oridiff_i'] = np.mean(oridiff_i)
    message['num_n'] = np.mean(num_n)
    message['num_i'] = np.mean(num_i)

    np.save(filename, message)
    messenger(message)

def high_dof(iter):

    chain7 = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    chain14 = Chain.from_urdf_file('r14.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False])
    chain21 = Chain.from_urdf_file('r21.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False])
    chain42 = Chain.from_urdf_file('r42.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False])
    chain_63 = Chain.from_urdf_file('r63.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False])
    # chain84 = Chain.from_urdf_file('r84.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False])

    chain = [chain7, chain14, chain21, chain42, chain_63]

    ax = [plot.figure().add_subplot(111, projection='3d') for _ in range(len(chain))]

    res = [-0.55, -0.55, -0.1, 0.55, 0.55, 0.9]

    dev = []
    time_n = []
    num_n = []
    stat_n = []
    time_i = []
    num_i = []
    stat_i = []

    for _ in range(iter):

        rp = np.array([r.uniform(-1, 1) for _ in range(3)])
        rp *= (0.02/np.linalg.norm(rp))

        ntime_tmp = []
        nnum_tmp = []
        nstat_tmp = []
        itime_tmp = []
        inum_tmp = []
        istat_tmp = []

        for i in range(len(chain)):
            # mul = 1
            mul = int((len(chain[i])-2)/7)

            # x = round(r.uniform(res[0]*mul, res[3]*mul), 4)
            # y = round(r.uniform(res[1]*mul, res[4]*mul), 4)
            # z = round(r.uniform(res[2]*mul, res[5]*mul), 4)
            # otarget = [x, y, z]
            # # print(otarget)
            # target = otarget + rp
            # # print(target)

            stat = 1
            while stat:
                radius = 0.7*mul
                x = round(r.uniform(-radius, radius), 4)
                radius -= x
                y = round(r.uniform(-radius, radius), 4)
                radius -= y
                z = round(r.uniform(-radius+0.6, radius+0.2), 4)
                otarget = [x, y, z]
                target = otarget + rp

                oresult, otime, stat, oj_list = chain[i].inverse_kinematics(otarget, initial_position=[0.0]*len(chain[i]))

            iresult, itime, istat, ij_list = chain[i].inverse_kinematics(target, initial_position=[0.0]*len(chain[i]))
            nresult, ntime, nstat, nj_list = chain[i].inverse_kinematics(target, initial_position=oresult)
            ntime_tmp.append(ntime)
            nnum_tmp.append(len(nj_list))
            itime_tmp.append(itime)
            inum_tmp.append(len(ij_list))
            if nstat == 2:
                print(i, chain[i].forward_kinematics(nresult)[:3,3], target, otarget)
            if istat == 2:
                print(i, chain[i].forward_kinematics(iresult)[:3,3], target)

            nstat_tmp.append(True if not nstat else False)
            istat_tmp.append(True if not istat else False)

            # print(str(i)+'d')


            # stretch = [0.0, 0.0, 0.0, 0.0, 0.0, 3.14, 0.0]*mul
            # print([round(p, 4) for p in chain[i].forward_kinematics([0.0, *stretch, 0.0])[:3, 3]])

            # print(ni, ntime)
            # print(oi, otime)
            # print()
            # chain[i].plot([0.0, *stretch, 0.0], ax[i])
            # chain[i].plot(oresult, ax[i])
            # chain[i].plot(iresult, ax[i])
            # chain[i].plot(nresult, ax[i])
            # ax[i].scatter3D(target[0], target[1], target[2], c='red')

        # plot.show()

        dev.append(np.linalg.norm(rp))
        time_n.append(ntime_tmp)
        num_n.append(nnum_tmp)
        stat_n.append(nstat_tmp)
        time_i.append(itime_tmp)
        num_i.append(inum_tmp)
        stat_i.append(istat_tmp)

    message = {}
    message['dev'] = np.mean(dev)
    for i, c in enumerate(chain):
        message[str(len(c)-2)+'success_n'] = len(np.array(stat_n).T[i][np.array(stat_n).T[i]])
        message[str(len(c)-2)+'success_i'] = len(np.array(stat_i).T[i][np.array(stat_i).T[i]])
        message[str(len(c)-2)+'time_n'] = np.array(time_n).T[i][np.array(stat_n).T[i]].mean()
        message[str(len(c)-2)+'time_i'] = np.array(time_i).T[i][np.array(stat_i).T[i]].mean()
        message[str(len(c)-2)+'num_n'] = np.array(num_n).T[i][np.array(stat_n).T[i]].mean()
        message[str(len(c)-2)+'num_i'] = np.array(num_i).T[i][np.array(stat_i).T[i]].mean()

    np.save(RESULT_FOLDER+'e4high_dof_'+str(iter), messenge)
    messenger(message)

if __name__ == '__main__':
    print('start')
    start = d.datetime.now()

    # dataset = 'rtree_30'
    dataset = 'rtree_20'
    # dataset = 'dense'
    # dataset = 'full_jointonly_8'

    # fully_covered(1, dataset)
    # current_ik_speed(100)
    # ik_iteration(10000)
    # posture_num(1)
    # draw('rtree_20', 'inter_300_post')
    query_time(dataset, 10000, 1e-4)
    # high_dof(1000)
    # secondary_compare(dataset, 2000, 1e-4)
    # draw_line()

    # message = np.load(RESULT_FOLDER+dataset+'/'+'compare_10000_1e-06'+'.npy', allow_pickle=True)
    # message = np.load(RESULT_FOLDER+'high_dof_10000'+'.npy', allow_pickle=True)
    # messenger(message.item())

    print('duration: ', d.datetime.now()-start)
    print('end')
