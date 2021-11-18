from controller import Supervisor, Robot, Node
import numpy as np
def enumarate_joints(robot:Robot, timestep:int=32, verbose:bool=False):
    motors = {}
    position_sensors = {}
    device_number = robot.getNumberOfDevices()
    for i in range(device_number):
        device = robot.getDeviceByIndex(i)
        node_name = device.getName()
        node_type = device.getNodeType()
        if node_type == Node.ROTATIONAL_MOTOR or node_type == Node.LINEAR_MOTOR:
            motors[node_name] = robot.getMotor(node_name)
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
            position_sensors[node_name] = robot.getPositionSensor(node_name)
            position_sensors[node_name].enable(timestep)
    return motors, position_sensors

def get_sample_generator(filename):
    positions = np.genfromtxt(filename, delimiter=',')
    for position in positions:
        yield position
    return

def set_action(key, current_joint, current_joint_sensor):
    if(key == -1):
        return current_joint, current_joint_sensor, 0
    key = chr(key)
    joint = current_joint
    joint_sensor = current_joint_sensor
    trend = 0
    if key in '1234567':
        joint = f'panda_joint{key}'
        joint_sensor = f'panda_joint{key}_sensor'
    elif key == 'Z':
        trend = 0.1
    elif key == 'X':
        trend = -0.1
    return joint, joint_sensor, trend
