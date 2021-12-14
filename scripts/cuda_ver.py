import os
import numpy as np
import math as m
import datetime as d
from collections import namedtuple, defaultdict
import copy as c
import random as r

from numba import jit

from ik_simulator import IKTable, Robot

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

@jit
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

@jit
def diff_cal(list_1, list_2):
    return m.sqrt(sum([(i - j)**2 for i, j in zip(list_1, list_2)]))

class IKSimulator:
    def __init__(self, algo='pure'):
        self.iktable = IKTable('raw_data_7j_1')
        self.robot = Robot()
        self.algo = algo

    def messenger(self, message):
        for k, v in message.items():
            print(k+'\t'+str(v))

    def fk(self, joints):
        return self.robot.fk_dh(joints)

    def diff_cal(self, list_1, list_2):
        if len(list_1) == len(list_2):
            return m.sqrt(sum([(i - j)**2 for i, j in zip(list_1, list_2)]))
        else:
            print('length of two lists must be equal')
            return 0

    def find(self, target_position):
        searching_area = self.iktable.searching_area(target_position)

        return searching_area

    def index2pos_jo(self, indices):
        pos_jo = namedtuple('pos_jo', ['position', 'joint'])
        target_space = []
        for index in indices:
            target_space.append(pos_jo(self.iktable.positions[index], self.iktable.joints[index]))

        return target_space

    def get_different_postures(self, target_space):

        #finding arm posture types
        threshold = 1.5
        nearby_postures = []
        for i_joint, value in enumerate(target_space):
            for i_type in nearby_postures:
                diff = self.diff_cal(i_type.joint, value.joint)
                if diff < threshold:
                    # nearby_postures.append(value)
                    break
            else:
                nearby_postures.append(value)

        return nearby_postures


    def get_posts(self, indices):
        ref_joint = [3,5,6]

        threshold = 0.03
        nearby_postures = []
        for i_pos in indices:
            posture = []
            for i_type in nearby_postures:
                for i_jo in ref_joint:
                    posture.append(self.diff_cal(self.iktable.all_posi[i_pos][i_jo], self.iktable.all_posi[i_type][i_jo]))

                if (np.array(posture) < threshold).all():
                    break
            else:
                nearby_postures.append(i_pos)

        # print(len(indices), len(nearby_postures))

        return nearby_postures


    def find_all_posture(self, target_pos):
        positions = self.iktable.searching_area(target_pos)
        # target_space = self.index2pos_jo(positions)
        # nearby_postures = self.get_different_postures(target_space)

        target_space = self.get_posts(positions)
        nearby_postures = self.index2pos_jo(target_space)


        start = d.datetime.now()
        posture, message = self.posture_iter_machine(nearby_postures, target_pos)
        # self.vector_portion(nearby_postures, target_pos)

        # self.messenger(message)

        end = d.datetime.now()

        message['total time: '] = end-start
        print(' total time: ', message['total time: '])


        # return 0
        # return posture
        return message

    @staticmethod
    @jit(nopython=True)
    def approximation(imitating_joint, target_pos, moving_joint=[i for i in range(6)]):
        # 0.007915
        # rad_offset = [(m.pi/180.0)*(0.5**i) for i in range(3)]  #[1, 0.5, 0.25] degree
        rad_offset = [(m.pi/180.0)*(0.5**i) for i in range(7)]  #[1, 0.5, 0.25] degree

        dh = np.array([[0.0,     0.0, 0.333,     0.0],\
                       [0.0,     0.0,   0.0, -m.pi/2],\
                       [0.0,     0.0, 0.316,  m.pi/2],\
                       [0.0,  0.0825,   0.0,  m.pi/2],\
                       [0.0, -0.0825, 0.384, -m.pi/2],\
                       [0.0,     0.0,   0.0,  m.pi/2],\
                       [0.0,   0.088, 0.107,  m.pi/2]])
        dh[:,0] = imitating_joint

        fk_mat = np.eye(4)
        for i in range(7):
            dh_mat = np.array([[m.cos(dh[i,0])                    , -m.sin(dh[i,0])                    ,  0                  ,  dh[i,1]                    ],\
                      [m.sin(dh[i,0])*m.cos(dh[i,3]),  m.cos(dh[i,0])*m.cos(dh[i,3]), -m.sin(dh[i,3]), -dh[i,2]*m.sin(dh[i,3])],\
                      [m.sin(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,3]),  dh[i,2]*m.cos(dh[i,3])],\
                      [0                                      ,  0                                      ,  0                  ,  1                               ]])
            fk_mat = np.dot(fk_mat, dh_mat)
            # print(fk_mat[:3,3])

        diff = m.sqrt(sum([(i - j)**2 for i, j in zip(fk_mat[:3,3], target_pos)]))
        # print(diff)

        tmp_joint = imitating_joint

        for i in moving_joint:
            for offset in rad_offset:
                reverse = 0
                while reverse < 2:
                    tmp_joint[i] += offset
                    pre_diff = diff
                    dh = np.array([[0.0,     0.0, 0.333,     0.0],\
                                   [0.0,     0.0,   0.0, -m.pi/2],\
                                   [0.0,     0.0, 0.316,  m.pi/2],\
                                   [0.0,  0.0825,   0.0,  m.pi/2],\
                                   [0.0, -0.0825, 0.384, -m.pi/2],\
                                   [0.0,     0.0,   0.0,  m.pi/2],\
                                   [0.0,   0.088, 0.107,  m.pi/2]])
                    dh[:,0] = tmp_joint

                    fk_mat = np.eye(4)
                    for i in range(7):
                        dh_mat = np.array([[m.cos(dh[i,0])                    , -m.sin(dh[i,0])                    ,  0                  ,  dh[i,1]                    ],\
                                  [m.sin(dh[i,0])*m.cos(dh[i,3]),  m.cos(dh[i,0])*m.cos(dh[i,3]), -m.sin(dh[i,3]), -dh[i,2]*m.sin(dh[i,3])],\
                                  [m.sin(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,0])*m.sin(dh[i,3]),  m.cos(dh[i,3]),  dh[i,2]*m.cos(dh[i,3])],\
                                  [0                                      ,  0                                      ,  0                  ,  1                               ]])
                        fk_mat = np.dot(fk_mat, dh_mat)
                        # print(fk_mat[:3,3])

                    tmp_pos = fk_mat[:3,3]
                    diff = m.sqrt(sum([(i - j)**2 for i, j in zip(tmp_pos, target_pos)]))
                    # print(tmp_pos, diff)
                    if diff >= pre_diff:
                        offset *= -1
                        reverse += 1

                tmp_joint[i] += offset
                # print('joint %s with %s done' %(i+1, offset))



        # return tmp_joint, pre_diff, end-start
        return tmp_joint, pre_diff

    def posture_iter_machine(self, nearby_postures, target_pos):
        n = 0.0
        movements = [[] for _ in range(7)]
        origin_diff =  []
        time = []
        jo_diff = namedtuple('jo_diff', ['joint', 'diff'])
        posture = []
        for p_type in nearby_postures:
            s = d.datetime.now()

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

            e = d.datetime.now()

            posture.append(jo_diff(tmp_joint, diff))
            time.append(e-s)

            origin_diff.append(self.diff_cal(p_type.position, target_pos))

            for i in range(7):
                movements[i].append(abs(p_type.joint[i]-tmp_joint[i]))

            if diff > 0.005:#0.5cm
                n += 1
                # origin_diff.append(diff)

            # break

        # print([p.diff for p in posture])
        message = {}
        message['target:'] = target_pos
        message['posture: '] = len(posture)
        message['origin diff: '] = np.mean(origin_diff)
        message['mean diff: '] = np.mean(np.array([p.diff for p in posture]))
        message['origin std: '] = np.std(np.array(origin_diff))
        message['std error: '] = np.std(np.array([p.diff for p in posture]))
        message['worst diff: '] = max([p.diff for p in posture])
        message['worst%: '] = n/len(posture)
        message['worse num: '] = n
        # message['origin diff:'] = np.sort(origin_diff)
        message['avg. time: '] = np.mean(np.array(time))
        for i in range(7):
            movements[i] = np.mean(movements[i]).tolist()
        message['movements: '] = movements

        return posture, message

    def pure_approx(self, joint, target_pos):
        #moving joint:[5,4], ..., [5,4,3,2,1,0], joint[6] does not affect position
        tmp_joint = c.deepcopy(joint)
        for i in range(4, -1, -1):
            moving_joint = [j for j in range(i, 6, 1)]
            tmp_joint, diff = self.approximation(tmp_joint, target_pos, moving_joint=moving_joint)
            # tmp_joint, diff = self.approximation(tmp_joint, target_pos)

        # tmp_joint, diff = self.approximation(tmp_joint, target_pos)

        return tmp_joint, diff

    @staticmethod
    @jit(nopython=True)
    def best3joints(moves, tmp_joint, target_pos):

        joint2move = 0
        max = 0
        for jo in range(6):
            j1 = c.deepcopy(tmp_joint)
            j2 = c.deepcopy(tmp_joint)
            j1[jo] += moves[jo]
            j2[jo] -= moves[jo]

            vec = [round((a-b), 6) for a,b in zip(fk_dh(j1), fk_dh(j2))]
            dotp = abs(np.dot(np.subtract(target_pos, fk_dh(tmp_joint)), vec))
            # print(dotp)
            if dotp > max:
                max = dotp
                joint2move = jo
        return joint2move

    def vector_portion_v1(self, p_type, target_pos):
        # moves = [0.00753605, 0.01589324, 0.0483029, 0.03467266, 0.71936916, 0.16382559, 0.0]
        moves = [(1 * m.pi/180)]*7  #0.017453292519943295
        threshold = 0.001

        tmp_joint = c.deepcopy(p_type.joint)
        diff = self.diff_cal(p_type.position, target_pos)
        # vectors = []
        loop = 0
        while diff > threshold and loop < 20:
            loop += 1
            joint2move = self.best3joints(moves, tmp_joint, target_pos)

            # print(joint2move)
            tmp_joint, diff = self.approximation(tmp_joint, target_pos, moving_joint=[joint2move])

        return tmp_joint, diff

    def vector_portion_v2(self, p_type, target_pos):
        # moves = np.array([0.00753605, 0.01589324, 0.0483029, 0.03467266, 0.71936916, 0.16382559, 0.0])
        # print(p_type)
        moves = np.array([(1 * m.pi/180)]*7)  #0.017453292519943295
        threshold = 0.001

        tmp_joint = c.deepcopy(p_type.joint)
        pre_diff = 1
        diff = self.diff_cal(target_pos, p_type.position)
        # print(diff)
        # tmp_joint, diff = self.approximation(tmp_joint, target_pos, moving_joint=[0])

        jmp = 0
        while jmp < 2 and abs(pre_diff-diff) > 0.001 and diff > threshold :
            joint2move = [[0,0] for _ in range(3)]
            vectors = [0]

            # joint 0
            tmp_joint, pre_diff = self.approximation(tmp_joint, target_pos, moving_joint=[0])

            # joint 1-5
            for jo in range(1, 6, 1):
                j1 = c.deepcopy(tmp_joint)
                j2 = c.deepcopy(tmp_joint)
                j1[jo] += moves[jo]
                j2[jo] -= moves[jo]

                vec = [round((a-b), 6) for a,b in zip(self.fk(j1),self.fk(j2))]
                vectors.append(vec)

                for i in range(3):
                    # dim_prop = abs(vec[i])/np.sum(np.absolute(vec[:i]+vec[i+1:]))
                    dim_prop = abs(vec[i])/np.sum(np.absolute(vec))
                    if dim_prop > joint2move[i][1]:
                        joint2move[i] = [jo, dim_prop]

            moving_mat = [vectors[dim] for dim, _ in joint2move]
            prop = np.dot(np.linalg.pinv(moving_mat), target_pos)
            moving_prop = [[d[0], p] for d, p in zip(joint2move, prop/np.max(prop))]
            # print(moving_prop)

            for i, p in moving_prop:
                tmp_joint[i] += p * moves[i]

            # pre_diff = diff
            diff = self.diff_cal(self.fk(tmp_joint), target_pos)
            if diff > pre_diff:
                jmp += 1
                for i, p in moving_prop:
                    tmp_joint[i] -= p * moves[i]
                moves /= 2.0
                # print(pre_diff)

            else:
                jmp = 0
                # print(diff)

            # break

        return tmp_joint, diff

def runner(ik_simulator, iter, filename):

    t20 = [[-0.0049, -0.6795, -0.3356], [-0.7539, 0.7033, 0.7446], [0.3505, -0.5509, -0.1833], [0.3189, 0.2082, -0.3363], [0.0107, 0.0137, -0.0184], [-0.4332, -0.2703, 0.4754], [0.1551, 0.76, 0.7149], [-0.8449, -0.114, 0.975], [-0.3954, -0.6995, 1.1191], [0.7958, -0.7913, 1.0731], [0.1736, -0.7634, 1.0207], [-0.1275, -0.7468, 0.5705], [-0.4789, 0.0986, 0.6545], [-0.3446, 0.1855, 0.5123], [-0.6056, -0.6849, -0.2469], [0.4778, 0.1782, 0.356], [0.2797, 0.2775, -0.2074], [0.0332, -0.2419, 0.1503], [-0.4058, -0.4729, 0.2572], [0.4637, -0.2291, 0.8074]]
    t50 = [[0.5758, -0.4885, 1.0271], [-0.6085, -0.4255, 0.5083], [0.2788, -0.095, 0.4679], [0.2401, 0.0485, 0.4424], [-0.5991, 0.0507, 0.076], [-0.7214, 0.8085, 1.0062], [0.6651, 0.2353, 0.1987], [0.5085, -0.7104, -0.0782], [0.1461, -0.5669, 0.7866], [0.3165, 0.7707, 0.2065], [0.2235, -0.2836, 0.5669], [-0.5145, 0.1317, -0.2905], [0.7966, -0.4236, 0.8411], [-0.3434, -0.1719, 0.942], [0.1089, 0.2281, 0.3716], [-0.7805, 0.017, 0.1633], [-0.2938, 0.6114, -0.1254], [0.6441, -0.465, 0.3824], [0.5869, 0.5077, 0.6117], [-0.7016, 0.7046, 0.0695], [0.6497, -0.6365, 0.1269], [-0.0412, -0.462, 0.3256], [0.3443, 0.2157, -0.2519], [-0.476, 0.2943, 0.1508], [-0.3097, -0.6039, 0.9085], [-0.5675, -0.1751, 1.1225], [-0.3002, -0.5436, 0.9165], [-0.0303, -0.1176, 0.5681], [0.0569, 0.7381, 0.4034], [0.0173, -0.4492, 0.2811], [0.5773, -0.6733, 0.3621], [0.1824, -0.2039, 0.6849], [0.2546, -0.6338, 0.0493], [0.1626, -0.5477, 0.326], [-0.7549, 0.1028, 0.6731], [0.0792, -0.631, 0.2127], [-0.5684, -0.7224, -0.303], [0.4818, -0.2907, -0.0176], [0.124, -0.2767, -0.3508], [0.7752, 0.018, 0.315], [-0.5902, 0.3012, 1.1607], [-0.8282, -0.4573, 0.9724], [-0.4213, -0.2136, 0.0478], [-0.4584, 0.7187, -0.2936], [0.7528, -0.0771, 0.5969], [-0.0336, -0.7186, 0.5271], [-0.7806, 0.6753, 0.0032], [-0.5677, -0.2493, 0.0197], [-0.5725, 0.3716, -0.1632], [-0.4644, -0.0062, 0.7572]]

    message = []
    # for i in range(iter):
    #     x = round(r.uniform(-0.855, 0.855), 4)
    #     y = round(r.uniform(-0.855, 0.855), 4)
    #     z = round(r.uniform(-0.36, 1.19), 4)
    #     target = [x, y, z]
    #     print(str(i+1)+': '+str(target))
    #     message.append(ik_simulator.find_all_posture(target))
    #     np.save('../data/result/'+filename, message)

    for i, target in enumerate(t20):
        print(str(i+1)+': '+str(target))
        message.append(ik_simulator.find_all_posture(target))

    np.save('../data/result/'+filename, message)
    print('done.')



def show_avg(ik_simulator, filename):
    message = np.load('../data/result/'+filename+'.npy', allow_pickle=True)

    mes = defaultdict(list)
    n = 0
    for m in message:
        for k, v in m.items():
            mes[k].append(v)

        if m['mean diff: '] <= 0.002:
            print(m['mean diff: '])
            n += 1
    print(n)
    print(len(message))

    result = {}
    # print(mes)
    for k, v in mes.items():
        if k == 'posture: ' or k == 'worse num: ' or k == 'worst diff: ' or k == 'avg. time: ' or k == 'total time: ':
            result[k] = np.mean(v, axis=0)
        else:
            result[k] = np.average(v, axis=0, weights=mes['posture: '])
        # print(v)

    # ik_simulator.messenger(result)

if __name__ == '__main__':
    print('start')
    # gather = DataCollection()
    # gather.without_colliding_detect('raw_data_7j_1')

    # table2 = IKTable('table3', 'raw_data_7j_1')
    # ik_simulator = IKSimulator()
    target = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]
    # ik_simulator.find_all_posture(target)

    s = d.datetime.now()
    runner(IKSimulator(algo='pure'), 500, 'result_pure')
    e = d.datetime.now()
    print('full process duration: ', e-s)

    # s = d.datetime.now()
    # runner(IKSimulator(algo='vp_v1'), 500, 'result_vp_v1')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)
    #
    # s = d.datetime.now()
    # runner(IKSimulator(algo='vp_v2'), 500, 'result_vp_v2')
    # e = d.datetime.now()
    # print('full process duration: ', e-s)

    # show_avg(ik_simulator, 'result_pure_v1')

    print('end')
