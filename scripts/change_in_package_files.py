

#ikpy\inverse_kinematics.py
#returning other infos
    import datetime
    start = datetime.datetime.now()
    res = scipy.optimize.minimize(target, optimize_total, chain.active_from_full(starting_nodes_angles), method=optimization_method, bounds=real_bounds, options=options)
    # res = scipy.optimize.minimize(target, optimize_total, chain.active_from_full(starting_nodes_angles), method=optimization_method, bounds=real_bounds, options=options)
    time = datetime.datetime.now() - start
    # print(res.nit)
    logs.logger.info("Inverse kinematic optimisation OK, done in {} iterations".format(res.nit))

    return chain.active_to_full(res.x, starting_nodes_angles), time, res.status, res.nit#, res.joint_list
    # return(np.append(starting_nodes_angles[:chain.first_active_joint], res.x))



#scipy\optimize\lbfgsb.py
#animation for robot movements
def ikpy_draw(joint_list, t):
    from ikpy.chain import Chain
    from ikpy.link import DHLink as Link

    chain = Chain.from_urdf_file('D:\Desktop\IK_simulator\scripts\panda_arm_hand_fixed.urdf', base_elements=['panda_link0'], active_links_mask=[False, True, True, True, True, True, True, True, False])
    # print(chain)

    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.855, 0.855])
    ax.set_ylim([-0.855, 0.855])
    ax.set_zlim([-0.36, 1.19])

    def init():
        return chain.plot([0.0, *joint_list[0], 0.0], ax)

    def update(i):
        # t = [0.0, 0.0, 0.0]
        ax.clear()
        ax.set_xlim([-0.855, 0.855])
        ax.set_ylim([-0.855, 0.855])
        ax.set_zlim([-0.36, 1.19])
        ax.scatter3D(t[0], t[1], t[2], c='red')
        diff = np.linalg.norm([p[3] for p in chain.forward_kinematics([0.0, *joint_list[i], 0.0])[:3]]-np.array(t))
        ax.text(0.5,0.5,2.1, str(i), fontsize=15)
        ax.text(0.5,0.5,2, str(diff), fontsize=15)
        ax.set_ylabel(str(t)+str(len(joint_list)), fontsize=15)
        return chain.plot([0.0, *joint_list[i], 0.0], ax)

    name = 'D:/Desktop/IK_simulator/data/result/move'
    k = 0
    filename = name+str(k)+'.gif'
    while os.path.exists(filename):
        k += 1
        filename = name+str(k)+'.gif'


    ani = FuncAnimation(fig, update, frames = len(joint_list), interval = 200, init_func=init, blit=False)
    # ani.save(filename, writer='imagemagick', fps=4)
    plt.show()
