import os
import numpy as np
import math as m
import datetime as d
from collections import namedtuple, defaultdict
import copy as c
import random as r

from numba import jit

from ik_simulator import IKTable, IKSimulator

from scipy.spatial import KDTree

DATA_FOLDER = '../data/'
RAW_DATA_FOLDER = DATA_FOLDER+'raw_data/'
TABLE_FOLDER = DATA_FOLDER+'table/'
# x = np.arange(100).reshape(10, 10)
#
# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
# def go_fast(a): # Function is compiled to machine code when called the first time
#     def aa(a):
#         trace = 0.0
#         for i in range(a.shape[0]):   # Numba likes loops
#             trace += np.tanh(a[i, i]) # Numba likes NumPy functions
#         return trace
#     trace = aa(a)
#     return a + trace              # Numba likes NumPy broadcasting
#
# print(go_fast(x))

@jit(nopython=True)
def fk_dh(joints):
    dh = np.array([[0.0,     0.0, 0.333,     0.0],\
                   [0.0,     0.0,   0.0, -m.pi/2],\
                   [0.0,     0.0, 0.316,  m.pi/2],\
                   [0.0,  0.0825,   0.0,  m.pi/2],\
                   [0.0, -0.0825, 0.384, -m.pi/2],\
                   [0.0,     0.0,   0.0,  m.pi/2],\
                   [0.0,   0.088, 0.107,  m.pi/2]])
    dh[:,0] = joints

    fk_mat = np.eye(4)
    for i in range(7):
        dh_mat = [[m.cos(dh[i,0])                    , -m.sin(dh[i,0])                    ,  0                  ,  dh[i,1]                    ],\
                  [m.sin(dh[i,0])*m.cos(dh[i,3]),  m.cos(dh[i,0])*m.cos(dh[i,3]), -m.sin(dh[i,3]), -dh[i,2]*m.sin(dh[i,3])],\
                  [m.sin(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,3]),  dh[i,2]*m.cos(dh[i,3])],\
                  [0                                      ,  0                                      ,  0                  ,  1                               ]]
        fk_mat = np.dot(fk_mat, dh_mat)
        # print(fk_mat[:3,3])

    return fk_mat[:3,3].tolist()

@jit(nopython=True)
def diff_cal(list_1, list_2):
    return m.sqrt(sum([(i - j)**2 for i, j in zip(list_1, list_2)]))

@jit
def posture_iter_machine(nearby_postures, target_pos):
    n = 0.0
    movements = [[] for _ in range(7)]
    origin_diff =  []
    time = []
    jo_diff = namedtuple('jo_diff', ['joint', 'diff'])
    posture = []
    for p_type in nearby_postures:
        # s = d.datetime.now()

        origin_d = self.diff_cal(p_type.position, target_pos)


        if self.algo == 'pure':
            tmp_joint, diff = self.pure_approx(p_type.joint, target_pos)
        elif self.algo == 'vp_v1':
            tmp_joint, diff = self.vector_portion_v1(p_type, target_pos)
        elif self.algo == 'vp_v2':
            tmp_joint, diff = self.vector_portion_v2(p_type, target_pos)
            tmp_joint, diff = self.pure_approx(tmp_joint, target_pos)
        else:
            tmp_joint, diff = self.pure_approx(p_type.joint, target_pos)

        for i in range(50):
            tmp_joint, diff = self.pure_approx(p_type.joint, target_pos)

        # e = d.datetime.now()

        posture.append(jo_diff(tmp_joint, diff))
        # time.append(e-s)

        origin_diff.append(origin_d)

        for i in range(7):
            movements[i].append(abs(p_type.joint[i]-tmp_joint[i]))

        if diff > 0.005:#0.5cm
            n += 1
            # origin_diff.append(diff)

        # break


    message = {}
    message['target'] = target_pos
    message['posture'] = len(posture)
    message['origin_diff'] = np.mean(origin_diff)
    message['mean_diff'] = np.mean(np.array([p.diff for p in posture]))
    message['origin_std'] = np.std(np.array(origin_diff))
    message['std_error'] = np.std(np.array([p.diff for p in posture]))
    message['worst_diff'] = max([p.diff for p in posture])
    message['worst%'] = n/len(posture)
    message['worse_num'] = n
    # message['origin diff:'] = np.sort(origin_diff)
    # message['avg. time'] = np.mean(np.array(time))
    for i in range(7):
        movements[i] = np.mean(movements[i]).tolist()
    message['movements'] = movements

    return posture, message

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
        data = table.searching_area(target)
        nearby_postures = ik_simulator.find(target)
        if result:
            message.append(ik_simulator.find_all_posture(target))
            np.save('../data/result/'+filename, message)

    # for i, target in enumerate(t20):
    #     print(str(i+1)+': '+str(target))
    #     message.append(ik_simulator.find_all_posture(target))

    np.save('../data/result/'+filename, message)
    print('done.')



def show_avg(ik_simulator, filename):
    message = np.load('../data/result/'+filename+'.npy', allow_pickle=True)

    mes = defaultdict(list)
    n = 0
    gdiff = []
    bdiff = []
    for m in message:
        if m['mean_diff'] > 0.05:
        # if True:
            n += 1
            for k, v in m.items():
                mes[k].append(v)

            # print(m['target:'], m['posture: '],  m['mean diff: '], m['origin diff: '])
            # print(m['posture: '], m['origin diff: ']-m['mean diff: '])
            gdiff.append(m['origin_diff']-m['mean_diff'])
        bdiff.append(m['origin_diff']-m['mean_diff'])

    print(gdiff)
    print(bdiff)
    print(n)
    print(len(message))

    result = {}
    # print(mes)
    for k, v in mes.items():
        # if k == 'target:':
        if k == 'target':
            continue
        # if k == 'posture: ' or k == 'worse num: ' or k == 'worst diff: ' or k == 'avg. time: ' or k == 'total time: ':
        if k == 'posture' or k == 'worse_num' or k == 'worst_diff' or k == 'avg. time' or k == 'total time':
            result[k] = np.mean(v, axis=0)
        else:
            # result[k] = np.average(v, axis=0, weights=mes['posture: '])
            result[k] = np.average(v, axis=0, weights=mes['posture'])
        # print(v)

    ik_simulator.messenger(result)

if __name__ == '__main__':
    print('start')
    # gather = DataCollection()
    # gather.without_colliding_detect('raw_data_7j_1')

    table = IKTable('raw_data_7j_1')
    # ik_simulator = IKSimulator()
    # target = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]
    target = [-0.8449, -0.114, 0.975]
    # data = table.searching_area(target)

    # ik_simulator.find_all_posture(target)


    s = d.datetime.now()
    # runner(IKSimulator(algo='pure'), 100, 'cuda_pure')
    print(diff_cal(np.array([1,2,3]), np.array([3,4,5])))
    e = d.datetime.now()
    print('full process duration: ', e-s)

    # s = d.datetime.now()
    # runner(IKSimulator(algo='vp_v1'), 100, '100_result_vp_v1')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)

    # s = d.datetime.now()
    # runner(IKSimulator(algo='vp_v2'), 100, '100_result_vp_v2')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)

    # ik_simulator = IKSimulator()
    # show_avg(ik_simulator, '100_result_pure')

    print('end')
