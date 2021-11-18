import os
import numpy as np
from controller import Robot, Node, Field, Supervisor

def joint_restriction(position:np.array):
    joints_q_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
    joints_q_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    # joints_q_max = [2.8973, 1.7628, 2.8973, 1.7628, 2.8973, 1.7628, 2.8973]
    # joints_q_min = [-2.8973, -1.7628, -2.8973, -1.7628, -2.8973, -1.7628, -2.8973]
    # for pos, q_max, q_min in zip(position, joints_q_max, joints_q_min):
    #     if pos > q_max or pos < q_min:
    #         print('Exceed joint ability', q_min, pos, q_max)
    #         return None
    restricted_pos = np.array([(pos if pos >= q_min else q_min) if pos <= q_max else q_max for pos, q_max, q_min in zip(position, joints_q_max, joints_q_min)])
    return restricted_pos
    # return position

def set_robot_name(robot_name:str) -> bool:
    os.environ['WEBOTS_ROBOT_NAME'] = robot_name
    return os.environ['WEBOTS_ROBOT_NAME'] == robot_name

def get_sample(filename)->list:
    positions = np.genfromtxt(filename, delimiter=',')
    for position in positions:
        yield position
    return

def get_joint_pos(motors:list, position_sensors:list)->np.array:
    return np.array([sensor.getValue() for sensor in position_sensors])

def set_joint_pos(motors:list, position_sensors:list, position:np.array)->np.array:
    position = joint_restriction(position)
    if position is not None:
        for motor, pos in zip(motors, position):
            motor.setPosition(pos)

def get_motor_config(robot:Robot, timestep:int=32, verbose:bool=False):
    motors = {}
    position_sensors = {}
    device_number = robot.getNumberOfDevices()
    for i in range(device_number):
        device = robot.getDeviceByIndex(i)
        node_name = device.getName()
        node_type = device.getNodeType()
        if node_type == Node.ROTATIONAL_MOTOR or node_type == Node.LINEAR_MOTOR:
            # motors[node_name] = robot.getMotor(node_name)
            motors[node_name] = robot.getDevice(node_name)
            if verbose:
                print(f'''
Joint       : {node_name}
Max Position: {motors[node_name].getMaxPosition()}
Min Position: {motors[node_name].getMinPosition()}
Max velocity: {motors[node_name].getMaxVelocity()}
Max Force   : {motors[node_name].getMaxForce()}
Max Torque  : {motors[node_name].getMaxTorque()}
                ''')
        elif node_type == Node.POSITION_SENSOR:
            # position_sensors[node_name] = robot.getPositionSensor(node_name)
            position_sensors[node_name] = robot.getDevice(node_name)
            position_sensors[node_name].enable(timestep)
    return [*motors.values()], [*position_sensors.values()]

def respown_object(supervisor:Supervisor, wbo:str, name:str=None, pos=None):
    root = supervisor.getRoot()
    children:Field = root.getField('children')
    children.importMFNode(-1, wbo)
    if name:
        node = supervisor.getFromDef(name)
        if pos is not None:
            trans_field:Field = node.getField('translation')
            trans_field.setSFVec3f(pos)
        return node
    else:
        return None
