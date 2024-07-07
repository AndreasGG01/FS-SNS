import pandas as pd
import numpy as np
from FeatureExtraction import *
import networkx as nx
# from Large_Graph_Extraction import *
from Large_Graph_Extraction_V2 import *

def graph_creation(file_path, output_path, output_edge_list, large_graph, edge_type = None, feature_file_path = None):
    if large_graph:
        G = large_graph_extraction_v2(file_path, output_path, output_edge_list)
        features_data = large_graph_feature_extraction(feature_file_path, output_edge_list)
        return G, features_data
    else:
        edgelist_df = pd.read_csv(file_path)
        
        if edge_type == 'matrix':
            print('matrix manipulation')
            edgelist_df = edgelist_df.astype(float)
            edgelist_df.values[[np.arange(len(edgelist_df))]*2] = np.nan
            edgelist_df = edgelist_df.stack().reset_index()
            edgelist_df.rename(columns = {'level_0': 'node1', 'level_1': 'node2', 0: 'edge'}, inplace = True)
            edgelist_df['node1'] = edgelist_df['node1'] + 1
            print(edgelist_df.columns)
            edgelist_df = edgelist_df[(edgelist_df['node2'] != 'Unnamed: 0') & (edgelist_df['edge'] != 0) & (edgelist_df['node2'] != edgelist_df['node1'])]
            edgelist_df.to_csv(output_edge_list)

        features_data = pd.read_csv(feature_file_path)

        G = nx.from_pandas_edgelist(edgelist_df, 'node1', 'node2', create_using= nx.Graph())
        list_edges = list(G.edges(data=True))
        nx.write_graphml(G, output_path)
        print('list_edges:', list_edges)
        return G, features_data