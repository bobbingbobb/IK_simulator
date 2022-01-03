import os
import numpy as np
import math as m
import datetime as d
import random as r

# from scipy.spatial import KDTree

from constants import *
from utilities import *
from data_gen import Robot, DataCollection
from ik_simulator import IKTable, IKSimulator

def runner(ik_simulator, iter, filename):

    t20 = [[-0.0049, -0.6795, -0.3356], [-0.7539, 0.7033, 0.7446], [0.3505, -0.5509, -0.1833], [0.3189, 0.2082, -0.3363], [0.0107, 0.0137, -0.0184], [-0.4332, -0.2703, 0.4754], [0.1551, 0.76, 0.7149], [-0.8449, -0.114, 0.975], [-0.3954, -0.6995, 1.1191], [0.7958, -0.7913, 1.0731], [0.1736, -0.7634, 1.0207], [-0.1275, -0.7468, 0.5705], [-0.4789, 0.0986, 0.6545], [-0.3446, 0.1855, 0.5123], [-0.6056, -0.6849, -0.2469], [0.4778, 0.1782, 0.356], [0.2797, 0.2775, -0.2074], [0.0332, -0.2419, 0.1503], [-0.4058, -0.4729, 0.2572], [0.4637, -0.2291, 0.8074]]
    t50 = [[0.5758, -0.4885, 1.0271], [-0.6085, -0.4255, 0.5083], [0.2788, -0.095, 0.4679], [0.2401, 0.0485, 0.4424], [-0.5991, 0.0507, 0.076], [-0.7214, 0.8085, 1.0062], [0.6651, 0.2353, 0.1987], [0.5085, -0.7104, -0.0782], [0.1461, -0.5669, 0.7866], [0.3165, 0.7707, 0.2065], [0.2235, -0.2836, 0.5669], [-0.5145, 0.1317, -0.2905], [0.7966, -0.4236, 0.8411], [-0.3434, -0.1719, 0.942], [0.1089, 0.2281, 0.3716], [-0.7805, 0.017, 0.1633], [-0.2938, 0.6114, -0.1254], [0.6441, -0.465, 0.3824], [0.5869, 0.5077, 0.6117], [-0.7016, 0.7046, 0.0695], [0.6497, -0.6365, 0.1269], [-0.0412, -0.462, 0.3256], [0.3443, 0.2157, -0.2519], [-0.476, 0.2943, 0.1508], [-0.3097, -0.6039, 0.9085], [-0.5675, -0.1751, 1.1225], [-0.3002, -0.5436, 0.9165], [-0.0303, -0.1176, 0.5681], [0.0569, 0.7381, 0.4034], [0.0173, -0.4492, 0.2811], [0.5773, -0.6733, 0.3621], [0.1824, -0.2039, 0.6849], [0.2546, -0.6338, 0.0493], [0.1626, -0.5477, 0.326], [-0.7549, 0.1028, 0.6731], [0.0792, -0.631, 0.2127], [-0.5684, -0.7224, -0.303], [0.4818, -0.2907, -0.0176], [0.124, -0.2767, -0.3508], [0.7752, 0.018, 0.315], [-0.5902, 0.3012, 1.1607], [-0.8282, -0.4573, 0.9724], [-0.4213, -0.2136, 0.0478], [-0.4584, 0.7187, -0.2936], [0.7528, -0.0771, 0.5969], [-0.0336, -0.7186, 0.5271], [-0.7806, 0.6753, 0.0032], [-0.5677, -0.2493, 0.0197], [-0.5725, 0.3716, -0.1632], [-0.4644, -0.0062, 0.7572]]

    message = []
    for i in range(iter):
        x = round(r.uniform(-0.855, 0.855), 4)
        y = round(r.uniform(-0.855, 0.855), 4)
        z = round(r.uniform(-0.36, 1.19), 4)
        target = [x, y, z]
        print(str(i+1)+': '+str(target))
        result = ik_simulator.find_all_posture(target)
        if result:
            message.append(result)
            np.save(RESULT_FOLDER+filename, message)

    # for i, target in enumerate(t20):
    #     print(str(i+1)+': '+str(target))
    #     message.append(ik_simulator.find_all_posture(target))

    np.save(RESULT_FOLDER+filename, message)
    print('done.')

def gather(scale, name):
    gather = DataCollection(scale=scale)
    gather.without_colliding_detect(name)

if __name__ == '__main__':
    print('start')
    start = d.datetime.now()

    # gather(20, 'raw_data_7j_20')

    # table = IKTable('raw_data_7j_30')
    target = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]
    # target = [-0.8449, -0.114, 0.975]

    # ik_simulator = IKSimulator(algo='vp_v2')
    # messenger(ik_simulator.find_all_posture(target))
    # print(ik_simulator.find(target)[0][0][6])


    # s = d.datetime.now()
    # runner(IKSimulator(algo='pure'), 300, 'test')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)
    #
    # s = d.datetime.now()
    # runner(IKSimulator(algo='vp_v1'), 300, '300_20near_result_vp_v1')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)

    # s = d.datetime.now()
    # runner(IKSimulator(algo='vp_v2'), 300, '300_003r_result_vp_v2')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)


    ik_simulator = IKSimulator()
    show_avg('300_003r_result_vp_v2')

    print('duration: ', d.datetime.now()-start)

    print('end')
