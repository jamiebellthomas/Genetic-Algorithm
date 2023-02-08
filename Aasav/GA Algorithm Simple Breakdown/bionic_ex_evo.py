import time
import random
import sys
import string
import numpy as np
import logging
import math
import scipy
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.spatial.distance import cdist, pdist, euclidean

import simulation.bsim as bsim
import behtree.treegen as tg
import behtree.tree_nodes as tree_nodes
import evo.evaluate as evaluate
import evo.operators as op

from matplotlib import animation, rc
from IPython.display import HTML


# Top level execution of evolutionary algorithm

if __name__ == '__main__':

	''' Required inputs to evolutionary algorithm:
		
		1. Create swarm object:
			- Define the swarm size.
			- Initialize with behavior 'none'.
			- Set agent speed.
			- Generate an array of agent positions.

		2. Create environment object:
			- Assign the environment object to the swarm.map attribute.

		3. Create the set of targets to search for:
			- Create a target set object and set it's state to match the environment.
			- Set the detection radius.
	'''

    # Set random seed
	#random.seed(1000)
	seed = 1000
	random.seed(seed)
	np.random.seed(seed)

	# Create swarm object
	swarmsize = 30
	swarm = bsim.swarm()
	swarm.size = swarmsize
	swarm.behaviour = 'none'
	swarm.speed = 0.5
	swarm.origin = np.array([0, 0])
	swarm.gen_agents()

	# Create environment
	env = bsim.map()
	'''
	Check that environment is set to the right one! 
	'''
	env.bounded = True
	env.map1() 
	env.gen()
	swarm.map = env
	
	targets = bsim.target_set()
	targets.set_state('set1')
	targets.radius = 1.5
	targets.reset()

	# The test duration for each search attempt
	timesteps = 500

	fields = []
	fitness_maps = []
	field, grid = bsim.potentialField_map(env)
	trials = 1

    ###############################################################################
	
    #                     EDIT THIS SECTION OF CODE ONLY

	'''
		EVOLUTIONARY PARAMETERS:
		NGEN - Number of evoltuionary generations.
		popsize - Number of individuals in each population.
		indsize - The maximum depth of trees that are initially generated.
		tournsize - The number of individuals taken in each tournament selection.
		mutrate - Probability of performing a mutation on a node of a tree.
		growrate - Probability of growing a random tree during mutation.
		growdepth - The depth of randomly grown trees.
		hallsize - The number of individuals saved in the hall of fame.
		elitesize - The number of the best individuals saved between generations.  
	'''

	NGEN = 4; popsize = 10; indsize = 2
	tournsize = 6; mutrate = 0.02; growrate = 0.4; growdepth = 2
	hall = []; hallsize = 20; newind = 0
	elitesize = 4
	treecost = 0.01

    # NOTE: keep the treecost < 0.01 and > 0.001

	'''
    The blackboard is a python dictionary which contains all the possible paramters
    which can be used to build our robot supervisor.

    For instance, the "actions" entry contains all the types of behaviours that the 
    robot can perform and allow the supervisor to change what the robots are doing.

    The "metrics" entry describe how the supervisor can see the swarm, here we 
    can look at the median position of all the robots and their density.
    Using these metrics we can build condition statements within the behaviour tree
    and manipulate using evolutionary mutation operators.
    '''

	blackboard = {"operators": ['sel','seq'], 
					"opsize": [2,3,4,5,6,7],
					"act_types": ['act','param'],
					"actions": ['disperse','north','south','west','east','southeast',
                                'southwest','northeast','northwest'],
					"dirparam": [10,30,60],
					"rotparam": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09],
					"randparam": [0.01,0.02,0.03],
					"envcontrol": ['attract','repel'],
					"envcontrol_x": [-10,-8,-6,-4,-2,0,2,4,6,8,10],
					"envcontrol_y": [-10,-8,-6,-4,-2,0,2,4,6,8,10],
					"metrics": ['medianx', 'mediany', 'density'],
					"dimx": [-30,-20,-16,-12,-8,-4,0,4,8,12,16,20,30],
					"dimy": [-30,-20,-16,-12,-8,-4,0,4,8,12,16,20,30],
					"density": [2,6,10,14,18,22,26],
					"coverage": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
					}

	
    ##################################################################################

	op_prob = 0.55
	selectionNum = popsize - elitesize - newind
	
	# Generate starting population
	pop = []
	generations = []*NGEN
	pop = [tg.individual(tg.tree().make_tree(indsize, blackboard)) for t in range(popsize)]
	
	#  Logging variables
	logfit = []; logpop = []; logavg = []; logmax = []
	stats = {"avgsize": [], "stdsize": [], "meanfit": [], "stdfit": [], "maxfit": []}
	
	# Start evolution!
	for i in range(0, NGEN):
		print ('GEN: ', i) 
		newpop = []		
		# Serial execution
		evaluate.serial_search(pop, swarm, targets, i, timesteps, treecost, field, grid)

		# Record results -------------------------------------------------------------------------------------------------
		pop.sort(key=lambda x: x.fitness, reverse=True)
		op.log(pop, hall, logpop, logfit, logavg, logmax, i)
		hall = op.hallfame(pop, hall[:], hallsize)
		
		# Generate the next population -----------------------------------------------------------------------------------
		elite = [ind.copy() for ind in pop[0:elitesize]]

		# Remove worst individuals
		newpop = op.tournament(pop[:], tournsize, selectionNum)
        # Perform crossover on selected individuals from tournament selection
		newpop = op.crossover(newpop[:], op_prob)
		# Finally perform mutation on all trees
		newpop = op.mutate(newpop[:], mutrate, growrate, growdepth, blackboard)

		pop = []
        # Add the elite individuals to the new population
		pop = list(newpop + elite)
  
print('\n\nEvolution has finished.\n\n')

print('\n The best solution had a fitness of %.2f.\n\n' % (hall[0].fitness))

# Plot the average and max fitness

fig, ax = plt.subplots()	
		
ax.plot(logavg, 'r-', label="Average Fitness")
ax.plot(logmax, 'g-', label="Maximimum Fitness")

ax.set_xlabel("Generation")
ax.set_ylabel("Fitness")
ax.set_ylim(bottom = 0)

legend = ax.legend(loc="upper left")

plt.show()

