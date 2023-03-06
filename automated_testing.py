import pandas as pd
# Open xlsx file
test_list = pd.read_excel('Tests_v1.xlsx')
print(test_list)
# Extract info from each row 
#Print Crossover type column
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
    fitness_sharing = test_details['Fitness Sharing']
    random_agents = test_details['Random Agents']
    print(id, env, num_generations, num_agents, network_size, mutation_rate, mutation_type, selection_type, selected_agents, crossover_type, crossover_rate, fitness_sharing, random_agents)