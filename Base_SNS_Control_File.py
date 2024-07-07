from Feature import *
from Network import * 
from FeatureExtraction import *
import time
from Evaluation import *
# from Evaluation_2 import *
from Graph_Creation import *
from FeatureSelection import *
from bayes_opt import BayesianOptimization
from hyperopt import fmin, tpe, hp

starttime = time.time()

#INITIAL PARAMETERS
#-------------------------------------------------------------------------------
# edge_file_path = "Test_Graphs\\musae_ENGB_edges.csv"
graph_path = "Test_Graphs\\HighTech_Managers_Advice.graphml"
# feature_file_path = "Test_Graphs\\deezer_europe_features.json"
feature_output_path = "Feature_Extraction\\HighTechFeatures.csv"
# output_edge_list = "Test_Graphs\\twitch_edges_1000.csv"
result_output = 'Test/MyNetwork/HighTech_network_wFeatures&Opt.csv'

file_type = 'graph'
large_graph = False
create_graph = False

#Set these parameters if feature extraction is not required (for large networks feature extraction will be dealt with separately)
feature_extraction_required = False
Feature_extraction_file_path = feature_output_path
feature_selection_required = False
final_results_output = pd.DataFrame(columns = ["Feature_comb","Density", "Modularity","Assortativity","Degree Distribution","Shortest Path Length Distribution","Clustering Coefficient Distribution"])

curr_results_output = []

#Variables for tracking the best simulation results
final_sim_data = None
best_degree_dist = None
opt_count = 0
#-------------------------------------------------------------------------------

#Set the features for the graph if they are not on the graph
if create_graph:
    G_target, features = graph_creation(edge_file_path, graph_path, output_edge_list, large_graph, feature_file_path = feature_file_path)
    print("Type of graph:", type(G_target))
    G_target = set_graph_features(G_target, features)
    G_target = nx.convert_node_labels_to_integers(G_target)
    nx.write_graphml(G_target, graph_path)

else:
    G_target_init = nx.read_graphml(graph_path)
    G_target = nx.convert_node_labels_to_integers(G_target_init)
    edge_num = G_target.number_of_edges()
    print('Number of nodes:', G_target.number_of_nodes())
    print('nodes of network:', G_target.nodes)
    print('Number of edges:', edge_num)
    nx.write_graphml(G_target, 'test_nodse')

features = featureExtraction(G_target, Feature_extraction_file_path, feature_extraction_required, file_type)
feature_columns = list(features.columns)
# print('Feature Columns:', feature_columns)
# if feature_selection_required:
#     feature_ranking_df, features_df = featureSelection(features_df = features)
#     features_ranked = feature_ranking_df['Feature'].to_list()

myfeats = Feature(features)
print('Number of nodes:', myfeats.numNodes)
print('nodes:', myfeats.nodes)
myfeats.CrispRepresentation()

np.random.seed(42)
# features = pd.DataFrame(np.random.uniform(0,1,30),columns=["f1"])

# print(myfeats.featureDifferenceDict)
#Set Optimisation parameters -----------------------------------------------------------------------
pDNA = pd.DataFrame(np.full([myfeats.numNodes,myfeats.numFeatures], 1),columns = myfeats.featureNames)
hDNA = pd.DataFrame(np.full([myfeats.numNodes,myfeats.numFeatures], -1),columns = myfeats.featureNames)

# pDNA = pd.DataFrame(np.random.uniform(0,1,30),columns = ['f1'])
# hDNA = pd.DataFrame(np.random.uniform(0,1,30),columns = ['f1'])


curr_results_output = []

final_sim_data = None
best_degree_dist = None
opt_count = 0
#---------------------------------------------------------------------------------------------------

def blackbox(args):
    
    global MyEval, sim_data_df, myNetwork, Degree_Distribution, opt_count, best_degree_dist, final_sim_data, optimized_pDNA, optimized_hDNA
    args_midpoint = int(len(args)/2)
    hargs = args[:args_midpoint]
    pargs = args[args_midpoint:]

    columns_list = pDNA.columns
    print("columns:",columns_list)
    for i in range(len(columns_list)):
        curr_H = hargs[i]
        curr_P = pargs[i]
        curr_column = columns_list[i]

        pDNA[curr_column] = curr_P
        hDNA[curr_column] = curr_H

    
    myDNA = SocialDNA(myfeats,pDNA=pDNA,hDNA=hDNA)

    myNetwork = Network(myfeats, myDNA, NumEdge=edge_num, randomInterference=0.001)
    myNetwork.Formation(randomseed=50)

    #Evaluate the simulated network with the target network
    
    print('Num_nodes:', myNetwork.Graph.number_of_nodes())

    try:
    # MyEval = NetworkEval(myNetwork.Graph,evalMetrics=["Node Degree","Node Clustering Coefficient","Shortest Path Length", "Density", "Modularity","Assortativity","Degree Distribution","Shortest Path Length Distribution","Clustering Coefficient Distribution"])

        MyEval = NetworkEval(myNetwork.Graph,evalMetrics=["Density", "Modularity","Assortativity", "Degree Distribution","Shortest Path Length Distribution","Clustering Coefficient Distribution"])

        MyEval.Similarity(G_target, numbins=100,zeroAlternative=10**(-5))
        sim_data_df = MyEval.SimilarityData
        Degree_Distribution = sim_data_df[sim_data_df['Metric'] == 'Degree Distribution']['Distance'].item()

        print("Degree_Distribution:", Degree_Distribution)

        if opt_count == 0:
            best_degree_dist = Degree_Distribution 
            final_sim_data = sim_data_df
            optimized_pDNA = pDNA
            optimized_hDNA = hDNA
        
        if Degree_Distribution < best_degree_dist:
            best_degree_dist = Degree_Distribution 
            final_sim_data = sim_data_df
            optimized_pDNA = pDNA
            optimized_hDNA = hDNA
            # optimized_pDNA.to_excel('Test/MyNetwork/optimized_pDNA.xlsx')
            # optimized_hDNA.to_excel('Test/MyNetwork/optimized_hDNA.xlsx')

        opt_count += 1

        print("curr opt loop:", opt_count)
        print("Best_dist:", best_degree_dist)
        
    except Exception as e:
        print("Error thrown:", e)

    return Degree_Distribution

#Set the number of pBounds based on the number of features
# pBounds = {'H1':(-1,1),'H2':(-1,1),'H3':(-1,1),'H4':(-1,1),'H5':(-1,1),'H6':(-1,1),'H7':(-1,1),'H8':(-1,1),'H9':(-1,1),'H10':(-1,1),'H11':(-1,1),'H12':(-1,1),'H13':(-1,1),'H14':(-1,1),'H15':(-1,1),'H16':(-1,1),'H17':(-1,1),'H18':(-1,1),'H19':(-1,1),'H20':(-1,1),'H21':(-1,1),'H22':(-1,1),'H23':(-1,1), 'H24':(-1,1),'P1':(-1,1),'P2':(-1,1),'P3':(-1,1),'P4':(-1,1),'P5':(-1,1),'P6':(-1,1),'P7':(-1,1),'P8':(-1,1),'P9':(-1,1),'P10':(-1,1),'P11':(-1,1),'P12':(-1,1),'P13':(-1,1),'P14':(-1,1),'P15':(-1,1),'P16':(-1,1),'P17':(-1,1),'P18':(-1,1),'P19':(-1,1),'P20':(-1,1),'P21':(-1,1),'P22':(-1,1),'P23':(-1,1), 'P24':(-1,1),}
# define a search space
# h_space = [hp.uniform('H1', -1,1),hp.uniform('H2', -1,1),hp.uniform('H3', -1,1),
#             hp.uniform('H4', -1,1), hp.uniform('H5', -1,1),hp.uniform('H6', -1,1),hp.uniform('H7', -1,1),
#             hp.uniform('H8', -1,1),hp.uniform('H9', -1,1),hp.uniform('H10', -1,1),
#             hp.uniform('H11', -1,1), hp.uniform('H12', -1,1),hp.uniform('H13', -1,1),hp.uniform('H14', -1,1),
#             hp.uniform('H15', -1,1),hp.uniform('H16', -1,1),hp.uniform('H17', -1,1),
#             hp.uniform('H18', -1,1),hp.uniform('H19', -1,1),hp.uniform('H20', -1,1),
#             hp.uniform('H21', -1,1), hp.uniform('H22', -1,1),hp.uniform('H23', -1,1),hp.uniform('H24', -1,1)]
h_space = [hp.uniform('H1', -1,1),hp.uniform('H2', -1,1),hp.uniform('H3', -1,1),
            hp.uniform('H4', -1,1), hp.uniform('H5', -1,1),hp.uniform('H6', -1,1),hp.uniform('H7', -1,1), hp.uniform('H8', -1,1),hp.uniform('H9', -1,1),hp.uniform('H10', -1,1),
            hp.uniform('H11', -1,1), hp.uniform('H12', -1,1),hp.uniform('H13', -1,1),hp.uniform('H14', -1,1),
            hp.uniform('H15', -1,1)]

# h_space_curr = h_space[:select_features_num]

# p_space = [hp.uniform('P1', -1,1),hp.uniform('P2', -1,1),hp.uniform('P3', -1,1),
#             hp.uniform('P4', -1,1), hp.uniform('P5', -1,1),hp.uniform('P6', -1,1),hp.uniform('P7', -1,1),
#             hp.uniform('P8', -1,1),hp.uniform('P9', -1,1),hp.uniform('P10', -1,1),
#             hp.uniform('P11', -1,1), hp.uniform('P12', -1,1),hp.uniform('P13', -1,1),hp.uniform('P14', -1,1),
#             hp.uniform('P15', -1,1),hp.uniform('P16', -1,1),hp.uniform('P17', -1,1),
#             hp.uniform('P18', -1,1),hp.uniform('P19', -1,1),hp.uniform('P20', -1,1),
#             hp.uniform('P21', -1,1), hp.uniform('P22', -1,1),hp.uniform('P23', -1,1),hp.uniform('P24', -1,1)]
p_space = [hp.uniform('P1', -1,1),hp.uniform('P2', -1,1),hp.uniform('P3', -1,1),
            hp.uniform('P4', -1,1), hp.uniform('P5', -1,1),hp.uniform('P6', -1,1),hp.uniform('P7', -1,1), hp.uniform('P8', -1,1),hp.uniform('P9', -1,1),hp.uniform('P10', -1,1),
            hp.uniform('P11', -1,1), hp.uniform('P12', -1,1),hp.uniform('P13', -1,1),hp.uniform('P14', -1,1),
            hp.uniform('P15', -1,1)]

# p_space_curr = p_space[:select_features_num]

space_curr = h_space + p_space
# minimize the objective over the space
best = fmin(blackbox, space_curr, algo=tpe.suggest, max_evals=20, rstate=np.random.default_rng(seed = 42))

timespentonsim = time.time()-starttime

#Print out results
print(final_sim_data)
print(type(final_sim_data), final_sim_data.dtypes)
print("Time:",timespentonsim)

myNetwork.writeHistory("Test/MyNetwork/")
final_sim_data.to_csv(result_output)