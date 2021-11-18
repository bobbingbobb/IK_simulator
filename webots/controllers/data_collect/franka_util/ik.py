from ikpy.chain import Chain
import os
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
# 1 0 0 x
# 0 0 1 y
# 0 -1 0 z
# 0 0 0 1
chain = Chain.from_urdf_file(f'{os.path.dirname(__file__)}/panda_arm_hand.urdf', base_elements=['panda_link0'], last_link_vector=[0, 0, 0.0584])#, active_links_mask=[False, True, True, True, True, True, True, True, False, False, False])
print(chain.links)
def get_ik(pos, initial_position=None, orientation=None):
    # if type(pos) != list:
    #     print('please input position as list')
    #     exit(0)
    # if orientation==None:
    #     orientation = -1
    print('IK orientation : ',orientation)
    if orientation==-1:
        return chain.inverse_kinematics(pos, [0, 0, -1], orientation_mode=None , initial_position=[0, *initial_position, 0, 0])[1:8]
    else:
        return chain.inverse_kinematics(pos, [orientation, 0, 0], orientation_mode='X', initial_position=[0, *initial_position, 0, 0])[1:8]
    # return chain.inverse_kinematics(pos, [0, 0, -1], orientation_mode='Z', initial_position=[0, *initial_position, 0, 0])[1:8]
    # return chain.inverse_kinematics(pos, [0, 0, -1], orientation_mode=None, initial_position=[0, *initial_position, 0, 0])[1:8]
    # return chain.inverse_kinematics(pos, initial_position=[0, *initial_position, 0, 0])[1:8]

def calculate_fk(joints):
    if type(joints) != list and len(joints) != 9:
        exit(0)
    return chain.forward_kinematics([0, *joints, 0, 0])

if __name__ == '__main__':
    target_vector = [p[3] for p in calculate_fk([0.000143811,-0.785328,-0.00028123,-2.35449,-0.000527306,1.5716,0.785101, 0, 0])[:3]]
    print(target_vector)
    # target_vector = [-0.08759189, -0.00385896, 0.92620779]
    joints = get_ik(target_vector)
    fig, ax = plot_utils.init_3d_figure()
    chain.plot([0, *joints, 0, 0, 0], ax, target=target_vector)

    plt.show()
