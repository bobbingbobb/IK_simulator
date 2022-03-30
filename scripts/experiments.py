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


def ik_speed(iter, dataset):
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])

    ik_simulator = IKSimulator(algo='ikpy', dataset=dataset)
    robot = Robot()
    property = index.Property(dimension=3, fill_factor=0.9)
    idx = index.Index(os.path.join(RAW_DATA_FOLDER, dataset), properties=property)

    filename = RESULT_FOLDER+'ikspeed_'+str(iter)

    qq = [[2.261452851613893, 1.2678638162730076, 0.5167773944514868, -0.8406070742899003, -0.10517209706492414, 1.1394790032667186, 0.0],\
          [2.4210027739579743, 1.3009966570761633, -1.42905396272644, -2.544041138299412, 0.9251163541744827, 2.6155393190302805, 0.0],\
          [-0.23549414942291902, 0.832423206187018, -2.647052510679955, -2.90748471768531, -1.9811254084542207, 2.6563330044414837, 0.0],\
          [-1.7102280285609228, 0.3044722893296248, -2.4956445713066047, -2.2429764453197514, -2.2437051753305766, 0.45326282452969463, 0.0],\
          [-0.6129173193870816, -1.1511205746000033, 2.6250537551939215, -2.979698403115334, -2.0753946826766727, 0.3023112367356362, 0.0],\
          [2.331212320860088, 0.3967371837919469, 2.0213161785110842, -0.6174748070755989, -1.6121575299591036, 1.2259739205661861, 0.0],\
          [-0.342772043964791, 0.32317127974206317, -1.3184599282339045, -0.9066461347123194, -1.32182111821633, 3.5273453253785036, 0.0],\
          [-1.1476877802361145, -0.3172329305030266, 0.7576410057568856, -1.3571365902173707, -2.6323875770767504, 2.109568969322213, 0.0],\
          [-0.7601296277880292, -1.6528577941914346, 1.1637214811895018, -2.7921499654865096, 2.3360360328300236, 1.9170155932734587, 0.0],\
          [-0.30282087908194955, 1.158792433208708, 0.5818125361881141, -1.1714247790177041, -2.606253919155665, 2.8541282406680093, 0.0]]

    threshold = 1e-4
    methods = ['L-BFGS-B', 'BFGS', 'SLSQP', 'CG', 'newton-cg']
    mes = {}
    for opm in methods[:4]:
        print(opm)
        ntime = []
        niteration = []
        time = []
        iteration = []
        otime = []
        oiteration = []
        # for q in qq:
        for i in range(iter):
            joints = []
            q = np.zeros(7)
            for j in range(6):
                q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)
            target_pos, target_ori = robot.fk_dh(q)

            joint = list(idx.nearest(target_pos.tolist(), objects='raw'))[0][1]

            result, nearby, nstat, nnum = chain.inverse_kinematics(target_pos, initial_position=[0.0, *joint, 0.0], optimization_method=opm)
            if np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target_pos)) < threshold:
                ntime.append(nearby)
                niteration.append(nnum)

            result, ikpy, stat, num = chain.inverse_kinematics(target_pos, initial_position=[0]*9, optimization_method=opm)
            if np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target_pos)) < threshold:
                time.append(ikpy)
                iteration.append(num)

            result, oikpy, ostat, onum = chain.inverse_kinematics(target_pos, target_ori, orientation_mode='Z', initial_position=[0]*9, optimization_method=opm)
            if np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target_pos)) < threshold:
                otime.append(oikpy)
                oiteration.append(onum)

        mes[opm+'_ntime'] = np.mean(ntime)
        mes[opm+'_time'] = np.mean(time)
        mes[opm+'_otime'] = np.mean(otime)
        mes[opm+'_niter'] = np.mean(niteration)
        mes[opm+'_iter'] = np.mean(iteration)
        mes[opm+'_oiter'] = np.mean(oiteration)

    np.save(filename, mes)
    messenger(mes)

def bout_data(iter, dataset):
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])

    ik_simulator = IKSimulator(algo='ikpy', dataset=dataset)
    robot = Robot()
    property = index.Property(dimension=3, fill_factor=0.9)
    idx = index.Index(os.path.join(RAW_DATA_FOLDER, dataset), properties=property)

    if dataset.startswith('rtree'):
        res = [-0.855, -0.855, -0.36, 0.855, 0.855, 1.19]
        if dataset.startswith('rtree_30'):
            dsf = 'rtree_30/'
            rang = 0.05
        elif dataset.startswith('rtree_20'):
            dsf = 'rtree_20/'
            rang = 0.03
        elif dataset.startswith('rtree_10'):
            dsf = 'rtree_10/'
            rang = 0.02
    elif dataset.startswith('dense'):
        res = [0.2, 0.45, 0.3, 0.25, 0.5, 0.35]
        rang = 0.001
        dsf = 'dense/'
    elif dataset.startswith('full'):
        res = [0.2, 0.4, 0.3, 0.215, 0.415, 0.315]
        rang = 0.0005
        dsf = 'full/'

    filename = RESULT_FOLDER+dsf+'post_num_'+str(iter)

    dev = []
    ee_dev = []
    query_num = []
    post_num = []
    success_num = []
    res_num = []

    for i in range(iter):
        # print(i)
        success = 0
        unique = 0
        q = np.zeros(7)
        for j in range(6):
            q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)
        # print(q)
        target_pos, target_ori = robot.fk_dh(q)

        pos_info = list(idx.intersection([t+offset for offset in (-rang, rang) for t in target_pos], objects='raw'))
        if len(pos_info) < 20:
            pos_info = list(idx.nearest(target_pos.tolist(), 50, objects='raw'))

        nearby_postures = ik_simulator.posture_comparison(pos_info)#index
        # nearby_postures = [pos_info[inds[0]] for inds in ik_simulator.posture_comparison_all_joint_sorted(pos_info)]#index

        query_num.append(len(pos_info))
        post_num.append(len(nearby_postures))

        for posture in nearby_postures:
            # tmp_joint, diff = ik_simulator.vector_portion_v2([posture[0][6], posture[1]], target_pos)
            # result, diff = ik_simulator.pure_approx(tmp_joint, target_pos)
            # tmp_pos, tmp_ori = robot.fk_jo(result)
            result, ti, stat, num = chain.inverse_kinematics(target_pos, initial_position=[0, *posture[1], 0])
            tmp_pos, tmp_ori = robot.fk_jo(result[1:8])
            diff = np.linalg.norm(tmp_pos[6]-target_pos)
            if diff < 1e-4:
                success += 1
                dev.append(np.linalg.norm(posture[0][6] - target_pos))
                ee_dev.append(np.dot(tmp_ori, posture[2]))
                if len(ik_simulator.posture_comparison([posture, [tmp_pos, result[1:8], tmp_ori]])) < 2:
                    unique += 1
        success_num.append(success)
        res_num.append(unique)

    mes = {}
    mes['dataset'] = dataset
    # mes['size'] = idx.get_size()
    mes['len'] = np.mean(dev)
    mes['worst'] = max(dev)
    mes['query_num'] = np.mean(query_num)
    mes['post_num'] = np.mean(post_num)
    mes['success_num'] = np.mean(success_num)
    mes['res_num'] = np.mean(res_num)
    mes['likeliness'] = np.mean(ee_dev)

    np.save(filename, mes)
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
            tmp_joint, time, stat, joint_list = chain.inverse_kinematics(target)
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

def ik_speed_draw(iter):
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
            tmp_joint, ntime, nstat, ni = chain.inverse_kinematics(target, initial_position=joint)

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

#abandon
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

def query_time(dataset, iter):
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)
    ik_simulator = IKSimulator(algo='ikpy', dataset=dataset)
    robot = Robot()
    property = index.Property(dimension=3, fill_factor=0.9)
    idx = index.Index(os.path.join(RAW_DATA_FOLDER, dataset), properties=property)

    if dataset.startswith('rtree'):
        res = [-0.855, -0.855, -0.36, 0.855, 0.855, 1.19]
        if dataset.startswith('rtree_30'):
            dsf = 'rtree_30/'
            rang = 0.05
        elif dataset.startswith('rtree_20'):
            dsf = 'rtree_20/'
            rang = 0.03
        elif dataset.startswith('rtree_10'):
            dsf = 'rtree_10/'
            rang = 0.02
    elif dataset.startswith('dense'):
        res = [0.2, 0.45, 0.3, 0.25, 0.5, 0.35]
        rang = 0.001
        dsf = 'dense/'
    elif dataset.startswith('full'):
        res = [0.2, 0.4, 0.3, 0.215, 0.415, 0.315]
        rang = 0.0005
        dsf = 'full/'

    filename = RESULT_FOLDER+dsf+'351querytime'+str(iter)
    print(dataset+'_'+str(iter))

    time_all = []
    pos_all = []
    time_300 = []
    time_50 = []
    time_1 = []

    for _ in range(iter):
        x = round(r.uniform(res[0], res[3]), 4)
        y = round(r.uniform(res[1], res[4]), 4)
        z = round(r.uniform(res[2], res[5]), 4)
        target = [x, y, z]

        # q = np.zeros(7)
        # for j in range(6):
        #     q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)
        # # print(q)
        # target = robot.fk_dh(q)[0].tolist()

        tar_coord = [t+offset for offset in (-rang, rang) for t in target]
        s = d.datetime.now()
        pos_info = idx.intersection(tar_coord)
        e = d.datetime.now()
        query_all = e-s
        pos_all.append(len(list(pos_info)))

        s = d.datetime.now()
        pos_info = idx.nearest(target, 300)
        e = d.datetime.now()
        query_300 = e - s

        s = d.datetime.now()
        pos_info = idx.nearest(target, 50)
        e = d.datetime.now()
        query_50 = e - s

        s = d.datetime.now()
        pos_info = idx.nearest(target, 1)
        e = d.datetime.now()
        query_1 = e - s

        time_all.append(query_all)
        time_300.append(query_300)
        time_50.append(query_50)
        time_1.append(query_1)


    message = {}
    message['query_all'] = np.mean(time_all)
    message['query_300'] = np.mean(time_300)
    message['query_50'] = np.mean(time_50)
    message['query_1'] = np.mean(time_1)
    message['pos_all'] = np.mean(pos_all)

    np.save(filename, message)
    messenger(message)

def secondary_compare(dataset, iter, threshold, pos_num):
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)
    robot = Robot()
    ik_simulator = IKSimulator(algo='ikpy', dataset=dataset)
    property = index.Property(dimension=3, fill_factor=0.9)
    idx = index.Index(os.path.join(RAW_DATA_FOLDER, dataset), properties=property)

    if dataset.startswith('rtree'):
        res = [-0.855, -0.855, -0.36, 0.855, 0.855, 1.19]
        if dataset.startswith('rtree_30'):
            dsf = 'rtree_30/'
            rang = 0.05
        elif dataset.startswith('rtree_20'):
            dsf = 'rtree_20/'
            rang = 0.03
        elif dataset.startswith('rtree_10'):
            dsf = 'rtree_10/'
            rang = 0.02
    elif dataset.startswith('dense'):
        res = [0.2, 0.45, 0.3, 0.25, 0.5, 0.35]
        rang = 0.001
        dsf = 'dense/'
    elif dataset.startswith('full'):
        res = [0.2, 0.4, 0.3, 0.215, 0.415, 0.315]
        rang = 0.0005
        dsf = 'full/'

    filename = RESULT_FOLDER+dsf+'secondary_'+str(iter)+'_'+str(pos_num)
    print(dataset+'_'+str(iter)+'_'+str(threshold))

    time_q = []
    time_c = []
    time_s = []
    query_num = []
    post_num = []

    time_n = []
    oridiff_n = []
    diff_n = []
    num_n = []
    ee_dev = []
    ori_dev = []

    time_i = []
    oridiff_i = []
    diff_i = []
    num_i = []

    time_no = []
    oridiff_no = []
    num_no = []

    qq = [[2.261452851613893, 1.2678638162730076, 0.5167773944514868, -0.8406070742899003, -0.10517209706492414, 1.1394790032667186, 0.0],\
          [2.4210027739579743, 1.3009966570761633, -1.42905396272644, -2.544041138299412, 0.9251163541744827, 2.6155393190302805, 0.0],\
          [-0.23549414942291902, 0.832423206187018, -2.647052510679955, -2.90748471768531, -1.9811254084542207, 2.6563330044414837, 0.0],\
          [-1.7102280285609228, 0.3044722893296248, -2.4956445713066047, -2.2429764453197514, -2.2437051753305766, 0.45326282452969463, 0.0],\
          [-0.6129173193870816, -1.1511205746000033, 2.6250537551939215, -2.979698403115334, -2.0753946826766727, 0.3023112367356362, 0.0],\
          [2.331212320860088, 0.3967371837919469, 2.0213161785110842, -0.6174748070755989, -1.6121575299591036, 1.2259739205661861, 0.0],\
          [-0.342772043964791, 0.32317127974206317, -1.3184599282339045, -0.9066461347123194, -1.32182111821633, 3.5273453253785036, 0.0],\
          [-1.1476877802361145, -0.3172329305030266, 0.7576410057568856, -1.3571365902173707, -2.6323875770767504, 2.109568969322213, 0.0],\
          [-0.7601296277880292, -1.6528577941914346, 1.1637214811895018, -2.7921499654865096, 2.3360360328300236, 1.9170155932734587, 0.0],\
          [-0.30282087908194955, 1.158792433208708, 0.5818125361881141, -1.1714247790177041, -2.606253919155665, 2.8541282406680093, 0.0]]

    methods = ['L-BFGS-B', 'BFGS', 'SLSQP', 'CG']
    opm = methods[0]

    # for q in qq:
    for _ in range(iter):
        q = np.zeros(7)
        for j in range(6):
            q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)
        target_pos, target_ori = robot.fk_dh(q)

        if pos_num == 'all':
            tar_coord = [t+offset for offset in (-rang, rang) for t in target_pos]
            s = d.datetime.now()
            pos_info = list(idx.intersection(tar_coord, objects='raw'))
            e = d.datetime.now()

            if len(pos_info) < 20:
                s = d.datetime.now()
                pos_info = list(idx.nearest(target_pos.tolist(), 50, objects='raw'))
                e = d.datetime.now()

            query = e - s
        else:
            s = d.datetime.now()
            pos_info = list(idx.nearest(target_pos.tolist(), pos_num, objects='raw'))
            e = d.datetime.now()
            query = e - s

        query_num.append(len(pos_info))

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

        result, nearby, ne_stat, ne_n = chain.inverse_kinematics(target_pos, initial_position=[0, *joint, 0], optimization_method='L-BFGS-B')
        # result, ne_n, nearby = chain.inverse_kinematics(target_pos, target_ori, orientation_mode='Z', initial_position=[0, *joint, 0])
        ne_oridiff = np.linalg.norm(ik_simulator.fk(joint)-np.array(target_pos))
        ne_pos, ne_ori = robot.fk_dh(result[1:8])
        ne_diff = np.linalg.norm(ne_pos-np.array(target_pos))

        result, nearori, no_stat, no_n = chain.inverse_kinematics(target_pos, target_ori, orientation_mode='Z', initial_position=[0, *joint, 0], optimization_method='L-BFGS-B')
        # no_oridiff = np.linalg.norm(ik_simulator.fk(joint)-np.array(target_pos))
        no_diff = np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target_pos))

        joint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result, ikpy, i_stat, i_n = chain.inverse_kinematics(target_pos, target_ori, orientation_mode='Z', initial_position=[0, *joint, 0], optimization_method=opm)
        i_oridiff = np.linalg.norm(ik_simulator.fk([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])-np.array(target_pos))
        i_diff = np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target_pos))

        if ne_diff < threshold:
            time_q.append(query)
            time_c.append(classify)
            time_s.append(outter_task)
            time_n.append(nearby)
            oridiff_n.append(ne_oridiff)
            diff_n.append(ne_diff)
            num_n.append(ne_n)
            ori_dev.append(ori_tmp)
            ee_dev.append(np.dot(target_ori, ne_ori))

        if no_diff < threshold:
            time_no.append(nearori)
            num_no.append(no_n)

        if i_diff < threshold:
            time_i.append(ikpy)
            oridiff_i.append(i_oridiff)
            diff_i.append(i_diff)
            num_i.append(i_n)

    idx.close()

    message = {}
    message['nearby'] = len(time_n)
    message['nearori'] = len(time_no)
    message['ikpy'] = len(time_i)
    message['query_num'] = np.mean(query_num)
    message['post_num'] = np.mean(post_num)
    message['likeliness'] = np.mean(ee_dev)
    message['olikeliness'] = np.mean(ori_dev)
    message['time_q'] = np.mean(time_q)
    message['time_c'] = np.mean(time_c)
    message['time_s'] = np.mean(time_s)
    message['time_n'] = np.mean(time_n)
    message['time_no'] = np.mean(time_no)
    message['time_i'] = np.mean(time_i)
    message['oridiff_n'] = np.mean(oridiff_n)
    message['oridiff_i'] = np.mean(oridiff_i)
    message['diff_n'] = np.mean(diff_n)
    message['diff_i'] = np.mean(diff_i)
    message['num_n'] = np.mean(num_n)
    message['num_no'] = np.mean(num_no)
    message['num_i'] = np.mean(num_i)

    # np.save(filename, message)
    messenger(message)

def accelerate(dataset, iter, threshold):
    chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)
    robot = Robot()
    ik_simulator = IKSimulator(algo='ikpy', dataset=dataset)
    property = index.Property(dimension=3, fill_factor=0.9)
    idx = index.Index(os.path.join(RAW_DATA_FOLDER, dataset), properties=property)

    if dataset.startswith('rtree'):
        res = [-0.855, -0.855, -0.36, 0.855, 0.855, 1.19]
        if dataset.startswith('rtree_30'):
            dsf = 'rtree_30/'
            rang = 0.05
        elif dataset.startswith('rtree_20'):
            dsf = 'rtree_20/'
            rang = 0.03
        elif dataset.startswith('rtree_10'):
            dsf = 'rtree_10/'
            rang = 0.02
    elif dataset.startswith('dense'):
        res = [0.2, 0.45, 0.3, 0.25, 0.5, 0.35]
        rang = 0.001
        dsf = 'dense/'
    elif dataset.startswith('full'):
        res = [0.2, 0.4, 0.3, 0.215, 0.415, 0.315]
        rang = 0.0005
        dsf = 'full/'

    filename = RESULT_FOLDER+dsf+'accelerate_'+str(iter)
    print(dataset+'_'+str(iter)+'_'+str(threshold))

    time_q = []

    time_n = []
    oridiff_n = []
    diff_n = []
    num_n = []

    time_i = []
    oridiff_i = []
    diff_i = []
    num_i = []

    time_r = []
    oridiff_r = []
    diff_r = []
    num_r = []

    origin = ik_simulator.fk([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q = np.zeros(7)
    for j in range(6):
        q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)

    for _ in range(iter):
        # print(q)
        target_pos, target_ori = robot.fk_dh(q)

        s = d.datetime.now()
        pos_info = list(idx.nearest(target_pos.tolist(), objects='raw'))
        e = d.datetime.now()
        query = e - s

        joint = pos_info[0][1]

        result, nearby, ne_stat, ne_n = chain.inverse_kinematics(target_pos, initial_position=[0, *joint, 0])
        # result, ne_n, nearby = chain.inverse_kinematics(target_pos, target_ori, orientation_mode='Z', initial_position=[0, *joint, 0])
        ne_oridiff = np.linalg.norm(ik_simulator.fk(joint)-np.array(target_pos))
        ne_pos, ne_ori = robot.fk_dh(result[1:8])
        ne_diff = np.linalg.norm(ne_pos-np.array(target_pos))

        joint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result, ikpy, i_stat, i_n = chain.inverse_kinematics(target_pos, initial_position=[0, *joint, 0])
        i_oridiff = np.linalg.norm(origin-np.array(target_pos))
        i_diff = np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target_pos))

        for j in range(6):
            q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)

        result, rand, r_stat, r_n = chain.inverse_kinematics(target_pos, initial_position=[0, *q, 0])
        r_oridiff = np.linalg.norm(ik_simulator.fk(q)-np.array(target_pos))
        r_diff = np.linalg.norm(ik_simulator.fk(result[1:8])-np.array(target_pos))

        if ne_diff < threshold:
            time_q.append(query)
            time_n.append(nearby)
            oridiff_n.append(ne_oridiff)
            diff_n.append(ne_diff)
            num_n.append(ne_n)

        if i_diff < threshold:
            time_i.append(ikpy)
            oridiff_i.append(i_oridiff)
            diff_i.append(i_diff)
            num_i.append(i_n)

        if r_diff < threshold:
            time_r.append(rand)
            oridiff_r.append(r_oridiff)
            diff_r.append(r_diff)
            num_r.append(r_n)

    idx.close()

    message = {}
    message['nearby'] = len(time_n)
    message['ikpy'] = len(time_i)
    message['rand'] = len(time_r)
    message['time_q'] = np.mean(time_q)
    message['time_n'] = np.mean(time_n)
    message['time_i'] = np.mean(time_i)
    message['time_r'] = np.mean(time_r)
    message['oridiff_n'] = np.mean(oridiff_n)
    message['oridiff_i'] = np.mean(oridiff_i)
    message['oridiff_r'] = np.mean(oridiff_r)
    message['diff_n'] = np.mean(diff_n)
    message['diff_i'] = np.mean(diff_i)
    message['diff_r'] = np.mean(diff_r)
    message['num_n'] = np.mean(num_n)
    message['num_i'] = np.mean(num_i)
    message['num_r'] = np.mean(num_r)

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

                oresult, otime, stat, o_n = chain[i].inverse_kinematics(otarget, initial_position=[0.0]*len(chain[i]))

            iresult, itime, istat, i_n = chain[i].inverse_kinematics(target, initial_position=[0.0]*len(chain[i]))
            nresult, ntime, nstat, n_n = chain[i].inverse_kinematics(target, initial_position=oresult)
            ntime_tmp.append(ntime)
            nnum_tmp.append(n_n)
            itime_tmp.append(itime)
            inum_tmp.append(i_n)
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

    dataset = 'rtree_30'
    # dataset = 'rtree_20'
    # dataset = 'rtree_10'
    # dataset = 'dense'
    # dataset = 'full_jointonly_8'


    # ds = ['rtree_30', 'rtree_20', 'rtree_10', 'dense', 'full_jointonly_8']
    # for dsf in ds[:3]:
    #     # query_time(dsf, 10000)
    #     bout_data(1000, dsf)


    # secondary_compare(dataset, 100, 1e-4, 'all')
    # secondary_compare(dataset, 1000, 1e-4, 500)
    # secondary_compare(dataset, 1000, 1e-4, 300)
    # secondary_compare(dataset, 1000, 1e-4, 50)

    # accelerate(dataset, 10000, 1e-4)

    # ik_speed(10000, dataset)
    # bout_data(1000, dataset)
    # query_time(dataset, 10000)

    # ik_speed_draw(100)
    # ik_iteration(10000)
    # posture_num(1)
    # draw('rtree_20', 'inter_300_post')
    # high_dof(1000)
    # secondary_compare(dataset, 1000, 1e-4)
    # draw_line()

    # robot = Robot()
    # q = np.zeros(7)
    # for _ in range(10):
    #     for j in range(6):
    #         q[j] = r.uniform(robot.joints[j].min, robot.joints[j].max)
    #     print(q.tolist())

    # message = np.load(RESULT_FOLDER+dataset+'/'+'secondary_10000_00001'+'.npy', allow_pickle=True)
    # message = np.load(RESULT_FOLDER+'ikspeed_1000'+'.npy', allow_pickle=True)
    # messenger(message.item())

    print('duration: ', d.datetime.now()-start)
    print('end')
