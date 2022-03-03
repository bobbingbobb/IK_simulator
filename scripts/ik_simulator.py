import os
import numpy as np
import math as m
import datetime as d
from collections import namedtuple, defaultdict
import copy as c
import random as r
from rtree import index

from constants import *
from utilities import *
from data_gen import Robot, DataCollection


class IKTable:
    def __init__(self, filename='raw_data_7j_30'):

        self.table = self._load_dataset()

        self.robot = Robot()
        self.range = 0.05

    def _load_dataset(self):
        print('loading...')
        start = d.datetime.now()

        p = index.Property(dimension=3)
        dataset = []
        for file in os.listdir(RAW_DATA_FOLDER):
            if file.startswith("tdense") and file.endswith(".dat"):
                dataset.append(index.Index(os.path.join(RAW_DATA_FOLDER, name_alignment(file)), properties=p))

        print('loaded. duration: ', d.datetime.now()-start)
        return dataset

    def query(self, target):
        target = pos_alignment(target)

        result = self.dot_query(target)
        if len(result) < 20:
            result = self.query_neighbor(target)

        # self.range = 0.01
        # result = self.query_neighbor(target)

        return result

    def query_neighbor(self, target):
        # return a list of position indices
        target = pos_alignment(target)

        target_space = self.rtree_query(target)
        count = 0
        while len(target_space) < 21 and count < 20:
            print('sparse!')
            if not self.pos_info_extension(target_space):
                return 0
            target_space = self.rtree_query(target)
            count += 20
            # return target_space

        # print(len(target_space))
        neighbor_space = self.neighbor_check(target, target_space)
        print(len(neighbor_space))

        if len(neighbor_space) > 100:
            print('dense!')
            # return neighbor_space[:100]

        return neighbor_space

    def neighbor_check(self, target, target_space):
        neighbor_space = []
        for ts in target_space:
            distance = np.linalg.norm(ts[0][6]-target)
            if distance < self.range:
                neighbor_space.append(ts)
            # else:
            #     print(distance, end=' ')

        return neighbor_space

    def dot_query(self, target):
        result = []
        for table in self.table:
            result += [item.object for item in table.nearest(c.deepcopy(target), 1, objects=True)]

        return result

    def rtree_query(self, target):
        # return [item.object for item in self.table.nearest(c.deepcopy(target), 20, objects=True)]
        range = self.range / 2.0
        # range = self.range

        result = []
        for table in self.table:
            result += [item.object for item in table.intersection([t+offset for offset in (-range, range) for t in target], objects=True)]

        if len(result) < 20:
            result = []
            for table in self.table:
                result += [item.object for item in table.nearest(c.deepcopy(target), 20, objects=True)]

        return result

    def insert(self, pos_info):
        self.table[0].insert(r.randint(0, 100000), pos_info[0][6].tolist(), obj=pos_info)

    def delete(self, target):
        pass

    def pos_info_extension(self, target_space):
        new_pos = []
        count = 0
        while len(new_pos) < 50 and count < 20:
            print(len(new_pos), end=' ')
            joint1 = target_space[r.randint(0, (len(target_space)-1))][1]
            joint2 = target_space[r.randint(0, (len(target_space)-1))][1]

            for j1, j2 in zip(joint1, joint2):
                if abs(j1-j2) > 3*m.pi/180:
                    avg_joint = [(j1 + j2)/2.0 for j1, j2 in zip(joint1, joint2)]

                    position, vec_ee = self.robot.fk_jo(avg_joint)
                    for p in position:
                        p = pos_alignment(p)
                    pos_info = (position, avg_joint, vec_ee)
                    self.insert(c.deepcopy(pos_info))
                    new_pos.append(pos_info)
                    count = 0
                    break
            count += 1
        print()

        # savep1 = [j[1] for j in target_space]
        # savep2 = [j[1] for j in new_pos]
        # np.save('tar', savep1)
        # np.save('new', savep2)

        return len(new_pos)
        # target_space.extend(new_pos)
        #
        # return nearby_postures


class IKSimulator:
    def __init__(self, algo='pure'):
        self.iktable = IKTable()
        # self.iktable = IKTable('dense')
        self.robot = Robot()
        self.algo = algo

        from ikpy.chain import Chain
        import ikpy.utils.plot as plot_utils
        self.chain = Chain.from_urdf_file('panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], last_link_vector=[0, 0, 0], active_links_mask=[False, True, True, True, True, True, True, True, False, False])

    def fk(self, joints, insert=False):
        if insert:
            return self.robot.fk_jo(joints)

        return self.robot.fk_dh(joints)[0]

    def find(self, target_pos):
        # return self.iktable.query(target_pos)
        pos_info = self.iktable.query(target_pos)
        if not pos_info:
            return 0
        print(len(pos_info))
        # nearby_postures = self.posture_comparison(pos_info)
        # nearby_postures = self.posture_comparison_all_joint(pos_info)#index
        nearby_postures = [[pos_info[i_type] for i_type in inds] for inds in self.posture_comparison_all_joint_sorted(pos_info)]#index

        # nearby_postures = []
        # nearby_postures.append(self.posture_comparison_all_joint(pos_info))
        # nearby_postures.append(self.posture_comparison_all_joint_sorted(pos_info))
        return nearby_postures

    def posture_comparison(self, pos_info):
        thres_3 = np.linalg.norm([0.316, 0.0825])/10.0#j1 - j3 range
        thres_5 = (thres_3 + np.linalg.norm([0.384, 0.0825]))/10.0#j3 - j5 range

        nearby_postures = []
        for i_pos in pos_info:
            for type in nearby_postures:
                if np.dot(i_pos[2], type[0][2]) > 0.8 and \
                   np.linalg.norm(i_pos[0][3]-type[0][0][3]) < thres_3 and \
                   np.linalg.norm(i_pos[0][5]-type[0][0][5]) < thres_5 and \
                   np.linalg.norm(i_pos[1][2]-type[0][1][2]) < 0.6:
                    type.append(i_pos)
                    break
                # if np.dot(i_pos[2], type[2]) > 0.9 and \
                #    np.linalg.norm(i_pos[0][3]-type[0][3]) < thres_3 and \
                #    np.linalg.norm(i_pos[0][5]-type[0][5]) < thres_5:
                #     break
            else:
                nearby_postures.append([i_pos])
                # nearby_postures.append(i_pos)

        # return [np[0] for np in nearby_postures]
        print(len(nearby_postures))
        return nearby_postures

    def posture_comparison_all_joint_sorted(self, target_space):
        thres = 0.5
        nearby_postures = []
        def sorting(sort_target, q):
            if len(sort_target) == 1:
                return [sort_target]
            result = []
            sort_ind = [sort_target[i] for i in np.argsort([target_space[st][1][q] for st in sort_target])]

            i = 0
            while True:
                r = 1
                if i+r == len(sort_ind):
                    result.append([sort_ind[i]])
                    return result
                while target_space[sort_ind[i+r]][1][q] - target_space[sort_ind[i]][1][q] < thres:#pure posture
                    r += 1
                    if not i+r < len(sort_ind):
                        break
                else:
                    if q == 5:
                        result.append(sort_ind[i:i+r])
                    else:
                        result += sorting(sort_ind[i:i+r], q+1)
                    i += r
                    continue

                if q == 5:
                    result.append(sort_ind[i:])
                    return result
                else:
                    result += sorting(sort_ind[i:], q+1)
                    break
            return result

        nearby_postures = sorting([_ for _ in range(len(target_space))], 0)

        return nearby_postures

    def posture_comparison_all_joint_sorted_pure(self, target_space):
        thres = 0.5
        nearby_postures = []
        def sorting(sort_target, q):
            if len(sort_target) == 1:
                return [sort_target]
            result = []
            sort_ind = [sort_target[i] for i in np.argsort([target_space[st][q] for st in sort_target])]#pure posture

            i = 0
            while True:
                r = 1
                if i+r == len(sort_ind):
                    result.append([sort_ind[i]])
                    return result
                while target_space[sort_ind[i+r]][q] - target_space[sort_ind[i]][q] < thres:#pure posture
                    r += 1
                    if not i+r < len(sort_ind):
                        break
                else:
                    if q == 5:
                        result.append(sort_ind[i:i+r])
                    else:
                        result += sorting(sort_ind[i:i+r], q+1)
                    i += r
                    continue

                if q == 5:
                    result.append(sort_ind[i:])
                    return result
                else:
                    result += sorting(sort_ind[i:], q+1)
                    break
            return result

        nearby_postures = sorting([_ for _ in range(len(target_space))], 0)

        return nearby_postures

    def posture_comparison_all_joint(self, target_space):
        thres = 0.5
        nearby_postures = []
        for pos in target_space:
            for type in nearby_postures:
                for j_pos, j_type in zip(pos[1], type[0][1]):
                    if abs(j_pos-j_type) >= thres:
                        break
                else:
                    type.append(pos)
                    break
            else:
                nearby_postures.append([pos])


        #index
        # for i_pos, v_pos in enumerate(target_space):
        #     # print(nearby_postures)
        #     for i_type, v_type in enumerate(nearby_postures):
        #         for j_pos, j_type in zip(v_pos[1], target_space[v_type[0]][1]):
        #             if abs(j_pos-j_type) >= thres:
        #                 break
        #         else:
        #             nearby_postures[i_type].append(i_pos)
        #             break
        #     else:
        #         nearby_postures.append([i_pos])

        return nearby_postures

    # abandoned
    def get_different_postures(self, target_space):
        #finding arm posture types
        threshold = 1.5
        nearby_postures = []
        for i_joint, value in enumerate(target_space):
            for i_type in nearby_postures:
                diff = np.linalg.norm(i_type.joint - value.joint)
                if diff < threshold:
                    # nearby_postures.append(value)
                    break
            else:
                nearby_postures.append(value)

        return nearby_postures

    # abandoned
    def index2pos_jo(self, indices):
        pos_jo = namedtuple('pos_jo', ['position', 'joint'])
        target_space = []
        for index in indices:
            # target_space.append(pos_jo(self.iktable.positions[index], self.iktable.joints[index]))
            target_space.append([self.iktable.all_posi[index], self.iktable.joints[index]])

        return target_space

    # abandoned
    def get_posts(self, indices):
        ref_joint = [3,5,6]

        threshold = 0.03
        nearby_postures = []
        for i_pos in indices:
            posture = []
            for i_type in nearby_postures:
                for i_jo in ref_joint:
                    # print(i_pos, i_type, i_jo)
                    posture.append(np.linalg.norm(self.iktable.all_posi[i_pos][i_jo] - self.iktable.all_posi[i_type][i_jo]))

                if (np.array(posture) < threshold).all():
                    break
            else:
                nearby_postures.append(i_pos)

        # print(len(indices), len(nearby_postures))

        return nearby_postures

    def find_all_posture(self, target_pos):
        start = d.datetime.now()

        nearby_postures = self.find(target_pos)
        if not nearby_postures:
            return 0
        posture, message = self.posture_iter_machine(nearby_postures, target_pos)
        # messenger(message)

        end = d.datetime.now()

        message['total time'] = end-start
        print(' total time: ', message['total time'])

        return message
        # return posture, message

    def posture_iter_machine(self, nearby_postures, target_pos, insert=False):
        n = 0.0
        movements = [[] for _ in range(7)]
        origin_diff =  []
        time = []
        jo_diff = namedtuple('jo_diff', ['joint', 'diff'])
        posture = []
        for p_type in nearby_postures:
            p_type = p_type[0]
            s = d.datetime.now()

            origin_d = np.linalg.norm(p_type[0][6] - target_pos)

            if self.algo == 'pure':
                tmp_joint, diff = self.pure_approx(p_type[1], target_pos)
            elif self.algo == 'vp_v1':
                tmp_joint, diff = self.vector_portion_v1([p_type[0][6], p_type[1]], target_pos)
            elif self.algo == 'vp_v2':
                tmp_joint, diff = self.vector_portion_v2([p_type[0][6], p_type[1]], target_pos)
                tmp_joint, diff = self.pure_approx(tmp_joint, target_pos)
            elif self.algo == 'ikpy':
                tmp_joint, diff = self.ikpy_run(p_type[1], target_pos)
            else:
                tmp_joint, diff = self.pure_approx(p_type[1], target_pos)

            # for i in range(50):
            #     tmp_joint, diff = self.pure_approx(p_type[1], target_pos)

            e = d.datetime.now()

            posture.append(jo_diff(tmp_joint, diff))
            # posture.append(tmp_joint)
            time.append(e-s)

            origin_diff.append(origin_d)

            for i in range(7):
                movements[i].append(abs(p_type[1][i]-tmp_joint[i]))

            if diff > 0.005:#0.5cm
                n += 1
                # origin_diff.append(diff)

            # break


        message = {}
        message['target'] = target_pos
        message['posture'] = len(posture)
        message['origin_diff'] = np.mean(origin_diff)
        message['mean_diff'] = np.mean(np.array([p.diff for p in posture]))
        # message['origin_std'] = np.std(np.array(origin_diff))
        # message['std_error'] = np.std(np.array([p.diff for p in posture]))
        message['worst_diff'] = max([p.diff for p in posture])
        message['worst%'] = n/len(posture)
        message['worse_num'] = n
        # message['origin diff:'] = np.sort(origin_diff)
        message['avg. time'] = np.mean(np.array(time))
        for i in range(7):
            movements[i] = np.mean(movements[i])
        message['movements'] = movements

        return posture, message

    def approximation(self, imitating_joint, target_pos, moving_joint=[i for i in range(6)]):
        # 0.007915
        start = d.datetime.now()

        rad_offset = [(m.pi/180.0)*(0.5**i) for i in range(3)]  #[1, 0.5, 0.25] degree
        # rad_offset = [(m.pi/180.0)*(0.5**i) for i in range(7)]  #[1, 0.5, 0.25] degree
        diff = np.linalg.norm(self.fk(imitating_joint) - target_pos)
        # print(diff)

        tmp_joint = imitating_joint

        for i in moving_joint:
            for offset in rad_offset:
                reverse = 0
                while reverse < 2:
                    tmp_joint[i] += offset
                    pre_diff = diff
                    tmp_pos = self.fk(tmp_joint)
                    diff = np.linalg.norm(tmp_pos - target_pos)
                    # print(tmp_pos, diff)
                    if diff >= pre_diff:
                        offset *= -1
                        reverse += 1

                tmp_joint[i] += offset
                # print('joint %s with %s done' %(i+1, offset))

        end = d.datetime.now()

        # return tmp_joint, pre_diff, end-start
        return tmp_joint, pre_diff

    def pure_approx(self, joint, target_pos):
        #moving joint:[5,4], ..., [5,4,3,2,1,0], joint[6] does not affect position
        tmp_joint = c.deepcopy(joint)
        for i in range(4, -1, -1):
            moving_joint = [j for j in range(i, 6, 1)]
            tmp_joint, diff = self.approximation(tmp_joint, target_pos, moving_joint=moving_joint)
            # tmp_joint, diff = self.approximation(tmp_joint, target_pos)

        # tmp_joint, diff = self.approximation(tmp_joint, target_pos)

        return tmp_joint, diff

    def vector_portion_v1(self, p_type, target_pos):
        # moves = [0.00753605, 0.01589324, 0.0483029, 0.03467266, 0.71936916, 0.16382559, 0.0]
        moves = [(1 * m.pi/180)]*7  #0.017453292519943295
        threshold = 0.001

        tmp_joint = c.deepcopy(p_type[1])
        diff = np.linalg.norm(p_type[0] - target_pos)
        # vectors = []
        loop = 0
        while diff > threshold and loop < 20:
            loop += 1
            joint2move = 0
            max = 0
            for jo in range(6):
                j1 = c.deepcopy(tmp_joint)
                j2 = c.deepcopy(tmp_joint)
                j1[jo] += moves[jo]
                j2[jo] -= moves[jo]

                vec = [round((a-b), 6) for a,b in zip(self.fk(j1),self.fk(j2))]
                dotp = abs(np.dot(np.subtract(target_pos, self.fk(tmp_joint)), vec))
                # print(dotp)
                if dotp > max:
                    max = dotp
                    joint2move = jo
            # print(joint2move)
            tmp_joint, diff = self.approximation(tmp_joint, target_pos, moving_joint=[joint2move])

        return tmp_joint, diff

    def vector_portion_v2(self, p_type, target_pos):
        # moves = np.array([0.00753605, 0.01589324, 0.0483029, 0.03467266, 0.71936916, 0.16382559, 0.0])
        # print(p_type)
        moves = np.array([(1 * m.pi/180)]*7)  #0.017453292519943295
        threshold = 0.001

        tmp_joint = c.deepcopy(p_type[1])
        pre_diff = 1
        diff = np.linalg.norm(target_pos - p_type[0])
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
            diff = np.linalg.norm(self.fk(tmp_joint) - target_pos)
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

    def ikpy_run(self, joint, target_pos):
        tmp_joint = self.chain.inverse_kinematics(target_pos, initial_position=[0, *joint, 0, 0])[1:8]
        diff = np.linalg.norm(self.fk(tmp_joint) - target_pos)

        return tmp_joint, diff

if __name__ == '__main__':
    print('start')
    start = d.datetime.now()

    # gather(20, 'raw_data_7j_20')
    target = [0.554499999999596, -2.7401472130806895e-17, 0.6245000000018803]
    # target = [-0.8449, -0.114, 0.975]

    # table = IKTable('raw_data_7j_30')
    # print(table.query_neighbor(target))

    ik_simulator = IKSimulator(algo='ikpy')
    print(ik_simulator.ikpy_run([0.0]*7, target))
    # result = ik_simulator.find(target)
    # print(result)
    # print(len(result[0]))
    # print()
    # print(len(result[1]))
    # for r in result:
    #     l = 0
    #     for i in r:
    #         l += len(i)
    #     print(l)

    # print(result)
    #26:5, 16,47:3
    # for i,v in enumerate(result):
    #     if len(v) > 3:
    #         print(len(v))
    # print([len(i) for i in result])
    # print([i[0][6] for i in result[4]])

    # print([jd.joint for jd in ik_simulator.find_all_posture(target)[0]])

    print('duration: ', d.datetime.now()-start)

    print('end')
