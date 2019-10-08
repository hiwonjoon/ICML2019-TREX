def gen_traj_dist(env,agent,render=False,max_len=99999):
    ob = env.reset()

    from mujoco_py.generated import const
    unwrapped = env
    while hasattr(unwrapped,'env'):
        unwrapped = unwrapped.env
    #viewer = unwrapped._get_viewer()
    #viewer.cam.fixedcamid = 0
    #viewer.cam.type = const.CAMERA_FIXED

    for _ in range(max_len):
        a = agent.act(ob,None,None)
        ob, r, done, _ = env.step(a)

    #    if render:
    #        env.render()
        if done:
            break

    return unwrapped.sim.data.qpos[0] #Final location of the object


def gen_traj_return(env,agent,render=False,max_len=99999):
    ob = env.reset()

    from mujoco_py.generated import const
    unwrapped = env
    while hasattr(unwrapped,'env'):
        unwrapped = unwrapped.env
    #viewer = unwrapped._get_viewer()
    #viewer.cam.fixedcamid = 0
    #viewer.cam.type = const.CAMERA_FIXED
    cum_return = 0.0
    for _ in range(max_len):
        a = agent.act(ob,None,None)
        ob, r, done, _ = env.step(a)
        cum_return += r
    #    if render:
    #        env.render()
        if done:
            break

    return cum_return
