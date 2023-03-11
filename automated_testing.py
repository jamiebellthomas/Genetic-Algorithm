import pandas as pd
import traceback
from GeneticAlgorithm import GeneticAlgorithm

# Open xlsx file
str_test_folder = 'RandomAgentTesting'
excel_filepath = 'RandomAgentTest.xlsx'
test_list = pd.read_excel(excel_filepath)

# Iterate through each test
for index, test_details in test_list.iterrows():
    id = test_details['Test ID']
    env = test_details['Environment']
    num_generations = test_details['Number of Generations']
    num_agents = test_details['Number of Agents']
    network_size = test_details['Network Size']
    mutation_rate = test_details['Mutation Rate']
    mutation_type = test_details['Mutation Type']
    selection_type = test_details['Selection Type']
    selected_agents = test_details['Selected Agents']
    crossover_type = test_details['Crossover Type']
    crossover_rate = test_details['Crossover Rate']
    random_agent_type = test_details['Random Agent Type']
    random_agent_rate = test_details['Random Agent Rate']
    fitness_sharing = test_details['Fitness Sharing']
    sparse_reward = test_details['Sparse Reward']
    description = test_details['Description']
    completed = test_details['Completed']

    # Check if test has already been completed
    if completed == 1:
        print('Test {} already been completed'.format(id))
        continue

    # Break if the test needs an entry from above
    if crossover_rate == 'BEST OF ABOVE':
        # Get the index of the best score up to the current index
        best_index = test_list.loc[:index, 'Final Mean Score'].idxmax()
        crossover_rate = test_list.loc[best_index, 'Crossover Rate']
        crossover_type = test_list.loc[best_index, 'Crossover Type']

        # Update test_list with the best crossover rate and type from current index to the last index
        test_list.loc[index:, 'Crossover Rate'] = crossover_rate
        test_list.loc[index:, 'Crossover Type'] = crossover_type

    if mutation_rate == 'BEST OF ABOVE':
        # Get the index of the best score up to the current index
        best_index = test_list.loc[:index, 'Final Mean Score'].idxmax()
        mutation_rate = test_list.loc[best_index, 'Mutation Rate']
        mutation_type = test_list.loc[best_index, 'Mutation Type']

        # Update test_list with the best mutation rate and type from current index to the last index
        test_list.loc[index:, 'Mutation Rate'] = mutation_rate
        test_list.loc[index:, 'Mutation Type'] = mutation_type

    if selected_agents == 'BEST OF ABOVE':
        # Get the index of the best score up to the current index
        best_index = test_list.loc[:index, 'Final Mean Score'].idxmax()
        selected_agents = test_list.loc[best_index, 'Selected Agents']
        selection_type = test_list.loc[best_index, 'Selection Type']

        # Update test_list with the best selected agents from current index to the last index
        test_list.loc[index:, 'Selected Agents'] = selected_agents
        test_list.loc[index:, 'Selection Type'] = selection_type

    if random_agent_rate == 'BEST OF ABOVE':
        # Get the index of the best score up to the current index
        best_index = test_list.loc[:index, 'Final Mean Score'].idxmax()
        random_agent_rate = test_list.loc[best_index, 'Random Agent Rate']
        random_agent_type = test_list.loc[best_index, 'Random Agent Type']

        # Update test_list with the best random agent rate and type from current index to the last index
        test_list.loc[index:, 'Random Agent Rate'] = random_agent_rate
        test_list.loc[index:, 'Random Agent Type'] = random_agent_type

    if fitness_sharing == 'BEST OF ABOVE':
        # Get the index of the best score up to the current index
        best_index = test_list.loc[:index, 'Final Mean Score'].idxmax()
        fitness_sharing = test_list.loc[best_index, 'Fitness Sharing']

        # Update test_list with the best fitness sharing from current index to the last index
        test_list.loc[index:, 'Fitness Sharing'] = fitness_sharing

    if mutation_type == 'None':
        mutation_type = 'random'

    if crossover_type == 'None':
        crossover_type = 'random'

    if random_agent_type == 'None':
        random_agent_type = 'fixed'

    # Run test
    # Instantiate Genetic Algorithm
    # Add try except and update test_list with error message and line number
    try:
        ga = GeneticAlgorithm(environment=env, population_size=num_agents, sparse_reward=sparse_reward, fitness_sharing=fitness_sharing,
                                num_select_agents=selected_agents, selection_type=selection_type, crossover_rate=crossover_rate, crossover_method=crossover_type,
                                mutation_rate=mutation_rate, mutation_method=mutation_type, num_generations=num_generations, parallel=False, plot=True,
                                settings=None, description=description, save_frequency=5, random_type=random_agent_type,
                                initial_random_rate=random_agent_rate, run_tests=True, str_test_folder=str_test_folder)

        # Run genetic algorithm
        ga.run()
    except Exception as e:
        # Save traceback to test_list
        test_list.loc[index, 'Error Message'] = traceback.format_exc()
        test_list.loc[index, 'Final Mean Score'] = 0
        test_list.loc[index, 'Final Max Score'] = 0
        test_list.loc[index, 'Total Runtime'] = 0
        test_list.loc[index, 'Average Runtime'] = 0
        print('Error with test {}'.format(id))

        # Save test_list
        test_list.to_excel(excel_filepath, index=False)
        continue

    # Update test_list
    test_list.loc[index, 'Completed'] = 1
    test_list.loc[index, 'Final Mean Score'] = ga.metrics['mean_fitness'][-1]
    test_list.loc[index, 'Final Max Score'] = max(ga.metrics['best_fitness'])
    test_list.loc[index, 'Total Runtime'] = ga.metrics['total duration'][-1]
    test_list.loc[index, 'Average Runtime'] = ga.metrics['total duration'][-1] / num_generations

    # Print the total runtime of the test script so far
    print('Total Runtime so far: {}'.format(test_list['Total Runtime'].sum()))

    # Save test_list
    test_list.to_excel(excel_filepath, index=False)
             