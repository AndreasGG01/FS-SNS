import pandas as pd
import numpy as np
import networkx as nx
pd.options.mode.chained_assignment = None

def featureExtraction(graph, filepath, extraction_required, file_type):
    if extraction_required:
        if file_type == 'text':
            feature_df = featureExtractionText(graph, filepath)
        elif file_type == 'graph':
            feature_df = featureExtractionGraph(graph, filepath)
        else:
            raise Exception("No file type given")
    else:
        feature_df = pd.read_csv(filepath, index_col=0)
    
    return feature_df

def featureExtractionGraph(feature_file, filepath):
    
    print("EXTRACTING GRAPH FEATURES...")
    feature_df = pd.DataFrame()
    list_of_dict_keys = []
    #Takes the first set of node feature keys (if nodes have different feature keys they shouldn't be used for the SNS)
    for n, a in feature_file.nodes(data = True, default = 0):
        list_of_dict_keys = list(feature_file.nodes[n].keys())
        if len(list_of_dict_keys) != 0:
            break
    
    print("List_of_keys:", list_of_dict_keys)
    num_keys = len(list_of_dict_keys)
    
    for i in range(num_keys):
        curr_key = list_of_dict_keys[i]
        locals()["key_" + curr_key] = []

    for n, a in feature_file.nodes(data = True, default = 0):
        curr_node = feature_file.nodes[n]
        if len(curr_node) != 0:
            for i in range(num_keys):
                curr_key = list_of_dict_keys[i]
                locals()["key_" + str(curr_key)].append(feature_file.nodes[n][curr_key])
        
    for i in range(num_keys):
        curr_key = list_of_dict_keys[i]
        feature_df['feature_' + str(curr_key)] = locals()["key_" + str(curr_key)]
    
    feature_df.to_csv(filepath)    
        
    return feature_df

def featureExtractionText(feature_file, filepath):
    feature_df = pd.read_csv(feature_file)
    test_df = feature_df[0]
    test_df.to_csv(filepath)

#For the graph creation set the features of the graph
def set_graph_features(G, features):
    features_dict = features.to_dict(orient='records')

    features_dict_final = {} 

    for row in features_dict:
        node = row.pop('nodes')
        features_dict_final.update({node: row})
    
    first_key = list(features_dict_final.keys())[0]
    first_value = features_dict_final[first_key]

    print("DICTIONRARY LENGHT",len(features_dict_final))
    print(first_key, first_value)
    nx.set_node_attributes(G, features_dict_final)

    return G

#Currently only works for Json
def large_graph_feature_extraction(feature_file_path, output_edge_list):
    edge_list_df = pd.read_csv(output_edge_list)

    node1_list = edge_list_df['node1'].unique().tolist()
    node2_list = edge_list_df['node2'].unique().tolist()
    node_list = node1_list + node2_list
    node_list = list(set(node_list))

    print(len(node_list))

    features_df = pd.DataFrame()
    final_features_df = pd.DataFrame()
    curr_df = pd.DataFrame()

    features_json_df = pd.read_json(feature_file_path, lines = True)
    column_list = features_json_df.columns
    print(features_json_df.shape[0])

    curr_list = features_json_df.loc[0, :].values.flatten().tolist()
    features_df['Feature_Col'] = curr_list

    features_df = pd.DataFrame(features_df.Feature_Col.values.tolist()).add_prefix('feature_')
    features_df.insert(0,'nodes', column_list)
    print(features_df.shape)

    features_df = features_df.iloc[:, :8]
    print(features_df.shape)

    checked_df = features_df[features_df['nodes'].isin(node_list)]
    # checked_df.dropna(inplace = True)
    # checked_df.dropna(axis='columns', inplace = True)
    checked_df['Is_null_check'] = checked_df.isnull().any(axis=1)
    nodes_to_remove_from_graph = checked_df[checked_df['Is_null_check'] == True][['nodes']]
    final_features_df = checked_df
    final_features_df = final_features_df.drop_duplicates()
    final_features_df.dropna(inplace = True)
    final_features_df.drop(columns = ['Is_null_check'], inplace = True)
    print(final_features_df.shape)

    final_features_df.to_csv('DeezerFeatures_check.csv')
    nodes_to_remove_from_graph.to_csv('nodes_to_remove_from_graph.csv')
    return final_features_df