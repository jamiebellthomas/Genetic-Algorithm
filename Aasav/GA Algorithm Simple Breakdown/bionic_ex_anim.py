    import random
    import sys
    import numpy as np
    import math
    import scipy
    import matplotlib.pyplot as plt
    from numpy.linalg import norm
    from scipy.spatial.distance import cdist, pdist, euclidean

    import simulation.bsim as bsim
    import simulation.environment as environment
    import behtree.treegen as tg
    import behtree.tree_nodes as tree_nodes
    import evo.evaluate as evaluate
    import evo.operators as op

    from matplotlib import animation, rc
    from IPython.display import HTML

    import py_trees
    import matplotlib.image as mpimg


    swarmsize = 30
    global swarm
    swarm = bsim.swarm()
    swarm.size = swarmsize
    swarm.speed = 0.5
    swarm.behaviour = 'random'
    swarm.param = 0.01
    swarm.gen_agents()

    targets = bsim.target_set()
    targets.set_state('set1')
    targets.radius = 3
    targets.reset()

    env = bsim.map()
    env.map1()
    env.gen()
    swarm.map = env

    field, grid = bsim.potentialField_map(env)
    swarm.field = field
    swarm.grid = grid

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(num=None, figsize=(8,8), dpi=100, facecolor='w', edgecolor='k')
    #plt.close() #Uncomment this line for collab

    dim = 52
    ax.set_xlim((-dim, dim))
    ax.set_ylim((-dim, dim))

    # Set how data is plotted within animation loop
    global line, target_found_line,target_unfound_line
    # Agent plotting 
    line, = ax.plot([], [], 'rh', markersize = 6, markeredgecolor="black", alpha = 0.9)
    # Box plotting
    target_found_line, = ax.plot([], [], 'gs', markersize = 8, markeredgecolor="black", alpha = 0.5)
    target_unfound_line, = ax.plot([], [], 'bs', markersize = 8, markeredgecolor="black", alpha = 0.5)
    #Text plotting
    plot_text = ax.text(0.5, 0.95,'',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes,size = 'large')


    line.set_data([], [])
    target_found_line.set_data([], [])
    target_unfound_line.set_data([], [])

    def init():
        target_found_line.set_data([], [])
        target_unfound_line.set_data([], [])
        plot_text.set_text('')
        return (line, target_found_line,target_unfound_line,plot_text)

    [ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
        [swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarm.map.obsticles))]

    timesteps = 500
    score = 0
    noise = np.random.uniform(-.01,.01,(timesteps, swarm.size, 2))

    #anim_type = 'evolved_swarm'
    anim_type = 'random_swarm'

    if anim_type == 'random_swarm':
        swarm.behaviour = 'random'
        swarm.param = 0.02
    elif anim_type == 'evolved_swarm':
        bt = tg.tree().decode(hall[0], swarm, targets)

    def animate(i): 
        if anim_type == "evolved_swarm":
              bt.tick()
        swarm.iterate(noise[i-1])
        swarm.get_state()
        score = targets.get_state_normal(swarm, i, timesteps)
        
        x = swarm.agents.T[0]
        y = swarm.agents.T[1]    
        line.set_data(x, y)

        unfound_targets = targets.targets[targets.old_state==False]
        a = unfound_targets.T[0]
        b = unfound_targets.T[1]
        target_unfound_line.set_data(a, b)

        found_targets = targets.targets[targets.old_state]
        a = found_targets.T[0]
        b = found_targets.T[1]
        target_found_line.set_data(a, b)

        plot_text.set_text("Targets found: {:d}/{:d}".format(score,len(targets.targets)))
        return (line, target_found_line,target_unfound_line,plot_text)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=timesteps, interval=200, blit=True)
    plt.show()
    # Note: below is the part which makes it work on Colab
    rc('animation', html='jshtml')
    anim
