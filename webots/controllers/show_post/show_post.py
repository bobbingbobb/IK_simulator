from controller import Supervisor, Node, RangeFinder
import os, subprocess, time
import numpy as np
import random as r
import math as m
import franka_util.utility as franka
import franka_util.ik as ik
import functools

from constants import *

object_list = []

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

def fk(position, joints:list):
    # [x, y, z, angle of the joint]
    jo = np.array([[    0.0,    0.0, 0.333,     0.0],\
                   [    0.0,    0.0,   0.0, -m.pi/2],\
                   [    0.0, -0.316,   0.0,  m.pi/2],\
                   [ 0.0825,    0.0,   0.0,  m.pi/2],\
                   [-0.0825,  0.384,   0.0, -m.pi/2],\
                   [    0.0,    0.0,   0.0,  m.pi/2],\
                   [  0.088,    0.0,   0.0,  m.pi/2]])
    cam = np.array([ 0.0424, -0.0424, 0.14, m.pi/4])# angle: dep_img coord to last joint
    gripper = np.array([ 0.0, 0.0, 0.107+0.0584+0.06, 0.0])
    flange = np.array([ 0.0, 0.0, 0.107, 0.0])

    position.append(1)
    fk_mat = np.eye(4)
    trans_mat = np.eye(4)
    #flange
    fk_mat = np.dot(rotate_z(flange[3]), fk_mat)
    for j in range(3):
        trans_mat[j,3] = flange[j]
    fk_mat = np.dot(trans_mat, fk_mat)
    #joints
    for i in range(6, -1, -1):
        fk_mat = np.dot(rotate_z(joints[i]), fk_mat)
        fk_mat = np.dot(rotate_x(jo[i, 3]), fk_mat)
        for j in range(3):
            trans_mat[j,3] = jo[i,j]
        fk_mat = np.dot(trans_mat, fk_mat)

    position = np.dot(fk_mat, position)
    p = [0, 0, 0, 1]
    print(np.dot(fk_mat, p))
    return position[:3]

def take_dep_img(name):
    image_name = name + '.png'
    npy_name = name + '.npy'
    # png_filename = IMG_FOLDER + image_name
    # npy_filename = IMG_FOLDER + npy_name

    if supervisor.step(timestep) != -1:
        range_finder.saveImage(image_name, 1)
        print(image_name+" saved.")
        dep = np.array(range_finder.getRangeImageArray())
        dep = np.transpose(dep)
        np.save(npy_name, dep)
        print(npy_name+" saved.")

def import_obj(object):
    name = object.split(".")
    object_list.insert(0, name[0])
    wbo_filename = WBO_FOLDER + name[0] + '.wbo'

    root = supervisor.getRoot()
    root_child = root.getField('children')
    root_child.importMFNode(-1 , wbo_filename)
    print('import ' + name[0])

    # random pose and rotation
    obj_solid = supervisor.getFromDef(name[0])
    obj_solid.resetPhysics()
    trans = obj_solid.getField('translation')
    rotate = obj_solid.getField('rotation')
    translation = [0, 0.78, 0] #on the table
    # translation = [r.uniform(-0.1, 0.1), r.uniform(0.8, 0.95), r.uniform(-0.1, 0.1)]
    # rotation = [r.random(), r.random(), r.random(), 3.1416]
    rotation = [0, 1, 0, r.uniform(-3.14, 3.14)]

    trans.setSFVec3f(translation)
    rotate.setSFRotation(rotation)

    # saving poses
    # sixd_pose_file = DATA_FOLDER + '6d_pose.txt'
    # f_6d = open(sixd_pose_file, 'a')
    # f_6d.write('%s: %f %f %f, %f %f %f\n' %(name[0], translation[0], translation[1], translation[2], rotation[0], rotation[1], rotation[2]))
    # f_6d.close()

def remove_obj(object):
    name = object.split(".")
    root = supervisor.getRoot()
    root_child = root.getField('children')


    if name[0] in object_list:
        obj_place = (-1) - object_list.index(name[0])
        root_child.removeMF(obj_place)
        object_list.remove(name[0])
        print(name[0] + ' removed')
    else:
        print('no such object.')

def get_current_pos(target_joints:list):
    # position: at the end of end effector
    fk = ik.calculate_fk(target_joints)
    # print(np.dot(fk,[0, 0, 0, 1]))
    return [p[3] for p in fk[:3]]

def obj_coord(object_name, dep_x, dep_y, dep_d):
    #compute object pose under robot coord

    #camear info
    height = range_finder.getHeight()
    width = range_finder.getWidth()
    h_fov = range_finder.getFov()
    # v_fov = 2 * m.atan(m.tan(h_fov * 0.5) * (height / width))

    max_wid = dep_d * m.tan(h_fov/2.0)
    unit = max_wid / (width / 2.0)

    #surface depth
    dep = np.load(IMG_FOLDER + object_name + '.npy')
    dep_surface = dep[dep_y][dep_x]

    # dep_cam(dep_image) coord
    x = (dep_x - width/2.0) * unit
    y = (dep_y - height/2.0) * unit
    # z = dep_d
    z = (dep_surface + dep_d) /2

    # print(x, y, z)target_pos = [0.6145-x, y, 0.5915-z]
    target_pos = [x, y, z]# under cam coord

    return target_pos

# def main(wbofile):
def main():
    global supervisor
    supervisor = Supervisor()
    global timestep
    timestep = int(supervisor.getBasicTimeStep())

    # print(supervisor.getFromDevice())

    motors, psensor = franka.get_motor_config(supervisor, timestep, verbose=False)
    # motors, psensor = franka.get_motor_config(supervisor.getFromDef('franka'), timestep, verbose=False)
    # motors_ex, psensor_ex = franka.get_motor_config(supervisor.getFromDef('franka_ex'), timestep, verbose=False)

    grip = 0.04
    origin_joints:list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    work_joints:list = [0.0, 0.0, 0.0, -1.57079632679, 0.0, 1.57079632679, 0.785398163397]
    # work_joints:list = [0.0, 1.5, 0.0, -2.5, 0.0, 1.57079632679, 0.785398163397]
    # work_joints:list = [0.2, 0.3, 0.0, -2.0, -0.4, 2.2, 0.785398163397]
    # work_joints:list = [1.1, -0.5, 0.1, -2.1, -1.3, 2.1, 0.7]
    max_joints:list = [3, 3, 3, 3, 3, 3, 3]
    min_joints:list = [-3, -3, -3, -3, -3, -3, -3]
    obj_data = [0.0, 0.0, 0.0, 0.0]

    rx = np.array([[1, 0, 0],\
                   [0, m.cos(-m.pi/2), -m.sin(-m.pi/2)],\
                   [0, m.sin(-m.pi/2), m.cos(-m.pi/2)]])
    rx_in = np.linalg.inv(rx)

    robot_arm = supervisor.getFromDef("franka")
    trans_field = robot_arm.getField('translation')
    robot_trans = trans_field.getSFVec3f()

    # ball_solid = supervisor.getFromDef("ball_target")
    # ball_trans = ball_solid.getField('translation')
    # ball_trans.setSFVec3f([0.7, 1, 0])

    jo_load = np.load('../../../scripts/new.npy', allow_pickle=True)
    jo_list = jo_load
    # jo_load = np.load('../../../scripts/example_eeonly.npy', allow_pickle=True)
    # jo_load = np.load('../../../scripts/example_disonly.npy', allow_pickle=True)
    # jo_list = [j[1] for j in jo_load[-2]]
    print(jo_list)

    # work_joints = [-0.7,  0.1,  0.8, -2.1, -2.8,  0.6,  0. ]

    # jo_list = [work_joints, [w+(2*m.pi/180) for w in work_joints]]

    # if supervisor.step(timestep) != -1:
    #     franka.set_joint_pos(motors, psensor, [*work_joints, grip, grip])
    count = 0
    jc = 0

    while supervisor.step(timestep) != -1:
        print(jc)
        print(jo_list[jc])

        count += 1
        if count == 50:
            # break
            count = 0
            franka.set_joint_pos(motors, psensor, [*jo_list[jc], grip, grip])
            jc += 1
            if jc == len(jo_list):
                break

    # print(franka.get_joint_pos(motors, psensor))
    # print(are_colliding(trans_field, trans_field))
    # print(trans_field)

    # print("ready")
    #
    # # saving poses
    # # sixd_pose_file = DATA_FOLDER + '6d_pose.txt'
    # # f_6d = open(sixd_pose_file, 'a')
    #
    # current_joints = franka.get_joint_pos(motors, psensor)
    # # for obj in wbofile:
    # if True:
    #     obj = 'NutCandy_800_tex'
    #     name = obj.split(".")
    #     obj_name = name[0]
    #     if supervisor.step(timestep) != -1:
    #         ball_trans.setSFVec3f([0.7, 1, 0])
    #     # f_6d.write('%s: ' %(obj_name))
    #     print(obj_name)
    #
    #     # import_obj(obj_name)
    #     take_dep_img(obj_name)
    #
    #     # depth_path = "/mnt/d/Users/Desktop/basic\ world\(currently\ used\)/data/images/depth_img/" + obj_name + ".npy"
    #     # f = open(COMM_FILENAME, 'w')
    #     # f.write(depth_path)
    #     # f.close()
    #
    #     print('waiting for GQCNN...')
    #     while supervisor.step(timestep) != -1:
    #         # f = open(COMM_FILENAME, 'r')
    #         # gqcnn_out = f.readline()
    #         # f.close()
    #
    #         # if not gqcnn_out.startswith('/'):
    #         if True:
    #             # if gqcnn_out == ',failed':
    #             if False:
    #                 f_6d.write('failed\n')
    #                 print('failed')
    #             else:
    #                 # data = gqcnn_out.split(',')
    #                 # obj_data[0] = int(float(data[1]))
    #                 # obj_data[1] = int(float(data[2]))
    #                 # obj_data[2] = float(data[3])
    #                 # obj_data[3] = float(data[4])
    #                 # print(obj_data)
    #                 obj_data = [320, 245, 0.547288, 1.489587]   # given by gqcnn
    #
    #                 target_pos = obj_coord(obj_name, obj_data[0], obj_data[1], obj_data[2])# target position under camera
    #                 target_pos = fk(target_pos, current_joints)# target position under world coord
    #                 # f_6d.write('%f, %f, %f, q=%f\n' %(target_pos[0], target_pos[1], target_pos[2], float(data[5])))
    #                 print("target position: %f, %f, %f"%(target_pos[0], target_pos[1], target_pos[2]))
    #
    #                 t = 0
    #                 while supervisor.step(timestep) != -1:
    #                     # move ball to target grasping position (just for testing)
    #                     ball_pos = np.dot(rx, target_pos)
    #                     ball_pos = ball_pos.tolist()
    #                     for i in range(len(ball_pos)):
    #                         ball_pos[i] += robot_trans[i]
    #                     ball_trans.setSFVec3f(ball_pos)
    #                     target_joints = ik.get_ik(target_pos, initial_position=current_joints, orientation=None)
    #                     print(target_joints)
    #                     franka.set_joint_pos(motors, psensor, [*target_joints, grip, grip])
    #                     t += 1
    #                     if t > 50:
    #                         break
    #
    #
    #                 # time.sleep(4)
    #                 # remove_obj(obj_name)
    #             break
    #     # break
    #
    #
    # # f_6d.close()
    # # while supervisor.step(timestep) != -1:
    # #     print('t')

if __name__ == '__main__':

    # list of object(.wbo)
    # wbofile = os.listdir(WBO_FOLDER)
    # wbofile = ['syringes5.wbo', 'syringes10.wbo', 'syringes20.wbo', 'syringes50.wbo']

    # main(wbofile)
    main()
