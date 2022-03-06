import os
import numpy as np
import datetime as d
import random as r
import copy as c

# from scipy.spatial import KDTree

from constants import *
from utilities import *
from data_gen import Robot, DataCollection
from ik_simulator import IKTable, IKSimulator

def runner(ik_simulator, iter, filename):

    t20 = [[-0.0049, -0.6795, -0.3356], [-0.7539, 0.7033, 0.7446], [0.3505, -0.5509, -0.1833], [0.3189, 0.2082, -0.3363], [0.0107, 0.0137, -0.0184], [-0.4332, -0.2703, 0.4754], [0.1551, 0.76, 0.7149], [-0.8449, -0.114, 0.975], [-0.3954, -0.6995, 1.1191], [0.7958, -0.7913, 1.0731], [0.1736, -0.7634, 1.0207], [-0.1275, -0.7468, 0.5705], [-0.4789, 0.0986, 0.6545], [-0.3446, 0.1855, 0.5123], [-0.6056, -0.6849, -0.2469], [0.4778, 0.1782, 0.356], [0.2797, 0.2775, -0.2074], [0.0332, -0.2419, 0.1503], [-0.4058, -0.4729, 0.2572], [0.4637, -0.2291, 0.8074]]
    t50 = [[0.5758, -0.4885, 1.0271], [-0.6085, -0.4255, 0.5083], [0.2788, -0.095, 0.4679], [0.2401, 0.0485, 0.4424], [-0.5991, 0.0507, 0.076], [-0.7214, 0.8085, 1.0062], [0.6651, 0.2353, 0.1987], [0.5085, -0.7104, -0.0782], [0.1461, -0.5669, 0.7866], [0.3165, 0.7707, 0.2065], [0.2235, -0.2836, 0.5669], [-0.5145, 0.1317, -0.2905], [0.7966, -0.4236, 0.8411], [-0.3434, -0.1719, 0.942], [0.1089, 0.2281, 0.3716], [-0.7805, 0.017, 0.1633], [-0.2938, 0.6114, -0.1254], [0.6441, -0.465, 0.3824], [0.5869, 0.5077, 0.6117], [-0.7016, 0.7046, 0.0695], [0.6497, -0.6365, 0.1269], [-0.0412, -0.462, 0.3256], [0.3443, 0.2157, -0.2519], [-0.476, 0.2943, 0.1508], [-0.3097, -0.6039, 0.9085], [-0.5675, -0.1751, 1.1225], [-0.3002, -0.5436, 0.9165], [-0.0303, -0.1176, 0.5681], [0.0569, 0.7381, 0.4034], [0.0173, -0.4492, 0.2811], [0.5773, -0.6733, 0.3621], [0.1824, -0.2039, 0.6849], [0.2546, -0.6338, 0.0493], [0.1626, -0.5477, 0.326], [-0.7549, 0.1028, 0.6731], [0.0792, -0.631, 0.2127], [-0.5684, -0.7224, -0.303], [0.4818, -0.2907, -0.0176], [0.124, -0.2767, -0.3508], [0.7752, 0.018, 0.315], [-0.5902, 0.3012, 1.1607], [-0.8282, -0.4573, 0.9724], [-0.4213, -0.2136, 0.0478], [-0.4584, 0.7187, -0.2936], [0.7528, -0.0771, 0.5969], [-0.0336, -0.7186, 0.5271], [-0.7806, 0.6753, 0.0032], [-0.5677, -0.2493, 0.0197], [-0.5725, 0.3716, -0.1632], [-0.4644, -0.0062, 0.7572]]

    message = []
    mes = {}
    for i in range(iter):
        # x = round(r.uniform(-0.855, 0.855), 4)
        # y = round(r.uniform(-0.855, 0.855), 4)
        # z = round(r.uniform(-0.36, 1.19), 4)
        x = round(r.uniform(0.2, 0.25), 4)
        y = round(r.uniform(0.45, 0.5), 4)
        z = round(r.uniform(0.3, 0.35), 4)
        target = [x, y, z]
        print(str(i+1)+': '+str(target))
        result = ik_simulator.find_all_posture(target)
        if result:
            message.append(result)
        else:
            mes['target'] = target
            mes['posture'] = 0
            message.append(c.deepcopy(mes))
        np.save(RESULT_FOLDER+filename, message)


    # for i, target in enumerate(t20):
    #     print(str(i+1)+': '+str(target))
    #     message.append(ik_simulator.find_all_posture(target))

    np.save(RESULT_FOLDER+filename, message)
    print('done.')

def sample_num(iter):
    ik_simulator = IKSimulator(algo='ikpy')
    num = []

    for i in range(iter):
        x = round(r.uniform(-0.855, 0.855), 4)
        y = round(r.uniform(-0.855, 0.855), 4)
        z = round(r.uniform(-0.36, 1.19), 4)
        target = [x, y, z]
        result = ik_simulator.find(target)
        if result:
            n = [0,0,0,0]
            for v in result:
                if len(v) ==1:
                    n[0] += 1
                elif len(v) ==2:
                    n[1] += 1
                elif len(v) == 3:
                    n[2] += 1
                else:
                    n[3] += 1

            n.append(len(result))
            num.append(n)

    num = np.array(num).T
    mes = {}
    mes['post_num'] = np.average(num[4])
    mes['1'] = np.average(num[0])
    mes['2'] = np.average(num[1])
    mes['3'] = np.average(num[2])
    mes['>4'] = np.average(num[3])

    messenger(mes)

def gather(scale, name):
    gather = DataCollection(scale=scale)
    gather.without_colliding_detect(name)

def change_test(iter):
    ik_simulator = IKSimulator(algo='ikpy')
    for i in range(iter):
        # x = round(r.uniform(-0.855, 0.855), 4)
        # y = round(r.uniform(-0.855, 0.855), 4)
        # z = round(r.uniform(-0.36, 1.19), 4)
        x = r.uniform(0.2, 0.25)
        y = r.uniform(0.45, 0.5)
        z = r.uniform(0.3, 0.35)
        target = [x, y, z]
        # before = ik_simulator.find(target)
        # print(before[10][1])
        after, _ = ik_simulator.find_all_posture(target)
        print(len(after), len(ik_simulator.posture_comparison_all_joint_sorted_pure([a.joint for a in after])))

if __name__ == '__main__':
    print('start')
    start = d.datetime.now()

    # gather(20, 'raw_data_7j_20')

    # table = IKTable('raw_data_7j_30')
    target = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]
    # target = [-0.8449, -0.114, 0.975]

    # ik_simulator = IKSimulator(algo='vp_v2')
    # messenger(ik_simulator.find_all_posture(target))
    # print(ik_simulator.find([-0.5906, 0.0, -0.1446]))
    # print(ik_simulator.find(target))


    # s = d.datetime.now()
    # runner(IKSimulator(algo='inter'), 300, 'inter_clean_300')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)
    #
    # s = d.datetime.now()
    # runner(IKSimulator(algo='pure'), 300, 'pure_clean_300')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)
    #
    # s = d.datetime.now()
    # runner(IKSimulator(algo='vp_v1'), 300, 'vp_v1_clean_300')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)
    #
    # s = d.datetime.now()
    # runner(IKSimulator(algo='vp_v2'), 300, 'vp_v2_clean_300')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)
    #
    # s = d.datetime.now()
    # runner(IKSimulator(algo='ikpy'), 300, 'ikpy_clean_300')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)

    # s = d.datetime.now()
    # runner(IKSimulator(algo='inter'), 100, 'test')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)

    ik_simulator = IKSimulator()
    show_avg('inter_clean_500')
    # show_avg('test')
    # show_avg('ikpy_100')
    # show_avg('100_006restrict_non_result_vp_v2')
    # show_avg('500_006restrict_result_vp_v2')
    # show_sparse('500_20plus_result_vp_v2')

    # ik_simulator = IKSimulator(algo='ikpy')
    # # target = [-0.8449, -0.114, 0.975]
    # posture, message = ik_simulator.find_all_posture(target)
    # print([jd.joint for jd in posture])
    # messenger(message)

    # sample_num(100)

    # change_test(1)

    print('duration: ', d.datetime.now()-start)

    print('end')
