import causalpy as cp
import numpy as np
import pandas as pd
import networkx as nx

def create_dot_graph(knowledge_dict, print_graph=True):
    """
    Generates a Graphviz dot graph representation of the knowledge structure defined in the `knowledge_dict` dictionary.
    
    Args:
        knowledge_dict (dict): A dictionary representing the knowledge structure, as parsed from a configuration file.
        print_graph (bool): Whether to print the graph or not.
    
    Returns:
        str: A string containing the Graphviz dot graph representation of the knowledge structure.
    """
    G = nx.DiGraph()
    edges = set()
    tiers = {}
    
    if 'knowledge' in knowledge_dict:
        for section in knowledge_dict['knowledge']:
            if section[0] == 'addtemporal':
                continue
            
            tier_num = section[0].rstrip('*')
            variables = section[1:]
            tiers[int(tier_num)] = variables
            G.add_nodes_from(variables)
            
            if section[0].endswith('*'):
                for var1 in variables:
                    for var2 in variables:
                        if var1 != var2:
                            G.add_edge(var1, var2, style='invis')
        
        for i in range(1, len(tiers)+1):
            for j in range(i+1, len(tiers)+1):
                for var_from in tiers[i]:
                    for var_to in tiers[j]:
                        edges.add((var_from, var_to))
                        G.add_edge(var_from, var_to)
    
    if 'forbiddirect' in knowledge_dict:
        for edge in knowledge_dict['forbiddirect']:
            if len(edge) == 2:
                G.add_edge(edge[0], edge[1], style='dashed', color='red', constraint='false')
    
    if 'requiredirect' in knowledge_dict:
        for edge in knowledge_dict['requiredirect']:
            if len(edge) == 2:
                G.add_edge(edge[0], edge[1])
    
    dot_graph = nx.drawing.nx_pydot.to_pydot(G).to_string()

    if print_graph:
        print(dot_graph)
    
    return dot_graph

def parse_knowledge_file(file_path):
    """
    Parses a knowledge file at the given file path and returns a dictionary of causal relations.
    
    Args:
        file_path (str): The path to the knowledge file to parse.
    
    Returns:
        dict: A dictionary of causal relations, where the keys are section names and the values are lists of causal relations.
    """
    causal_relations = {}
    current_section = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('/'):
                current_section = line[1:]
                causal_relations[current_section] = []
            elif line in ['forbiddirect', 'requiredirect']:
                current_section = line
                causal_relations[current_section] = []
            elif line and current_section:
                causal_relations[current_section].append(line.split())
    
    causal_relations = {k: v for k, v in causal_relations.items() if v}
    
    return causal_relations

def get_dataset_for_casualpy(dataName):
    """
    Retrieves a dataset for CausalPy analysis.
    
    Args:
        dataName (str): The name of the dataset to retrieve.
    
    Returns:
        dict: A dictionary containing the dataset and related information.
    """
    if dataName == 'iv':
        N = 100
        e1 = np.random.normal(0, 3, N)
        e2 = np.random.normal(0, 1, N)
        Z = np.random.uniform(0, 1, N)

        X = -1 + 4 * Z + e2 + 2 * e1
        y = 2 + 3 * X + 3 * e1

        return {
            'data': pd.DataFrame({"y": y, "X": X, "Z": Z}),
            'outcome': 'y',
            'treatment': 'X',
            'instruments': ['Z'],
        }
    
    return cp.load_data(dataName)