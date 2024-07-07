from Feature import *
from Network import * 
from FeatureExtraction import *
import time
from Evaluation import *
# from Evaluation_2 import *
from Graph_Creation import *
from FeatureSelection import *
from FeatureSelection_FSV2 import *
from FS_20240302 import *
from bayes_opt import BayesianOptimization
from hyperopt import fmin, tpe, hp

starttime = time.time()

#INITIAL PARAMETERS
#-------------------------------------------------------------------------------
file_type = 'graph'
large_graph = False
create_graph = False
network_name = 'Deezer'
date = '20240302'
FS_ver = 'FS_MutualInfo'
export_feature_selection = True

# edge_file_path = "Test_Graphs\\musae_ENGB_edges.csv"
graph_path = "Test_Graphs\\Deezer10k.graphml"
# feature_file_path = "Test_Graphs\\deezer_europe_features.json"
feature_output_path = "Feature_Extraction\\DeezerFeatures.csv"
# output_edge_list = "Test_Graphs\\HighTech_edges_processed.csv"
folder ='Simulation_Results_20240203/' + network_name + '_' + date
result_output = folder + '/' + network_name + '_' + FS_ver + '_SNS_' + date + '.csv'

#Set these parameters if feature extraction is not required (for large networks feature extraction will be dealt with separately)
feature_extraction_required = False
Feature_extraction_file_path = feature_output_path
feature_selection_required = True
final_results_output = pd.DataFrame(columns = ["Feature_comb","Density", "Modularity","Assortativity","Degree Distribution","Shortest Path Length Distribution","Clustering Coefficient Distribution", "Features Included"])
# final_results_output = pd.DataFrame(columns = ["Feature_comb","Density", "Modularity", "Assortativity", "Degree Distribution","Shortest Path Length Distribution","Clustering Coefficient Distribution","Node Degree", "Node Clustering Coefficient", "Node Pair Shortest Path Length"])
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
    # G_target_init.remove_node('Ant954')
    # G_target_init.remove_node('Ant665')
    edge_num = G_target.number_of_edges()
    print('Number of nodes:', G_target.number_of_nodes())
    print('nodes of network:', G_target.nodes)
    print('Number of edges:', edge_num)
    nx.write_graphml(G_target, 'test_nodse')

features = featureExtraction(G_target, Feature_extraction_file_path, feature_extraction_required, file_type)

feature_columns = list(features.columns)
# print('Feature Columns:', feature_columns)
if feature_selection_required:
    feature_ranking_df, features_df = FS_MutualInfo(features_df = features)
    features_ranked = feature_ranking_df['Feature'].to_list()
    if export_feature_selection:
        feature_ranking_df.to_excel('FS_SNS_Results\\' + FS_ver + '\\' + network_name + '_Feature_Ranking.xlsx')
        features_df.to_excel('FS_SNS_Results\\' + FS_ver + '\\' + network_name + '_Selection.xlsx')
else:
    features_ranked = features

# For feature combination rankings:
Overall_best_degree_dist = None

for i in range(len(features_ranked)):
    select_features_num = int(i + 1)
    print("Number of selection features:",select_features_num)
    selected_features = features_ranked[:select_features_num]

    print(selected_features)
    #Ensure the inputted features are in the same order
    l_replace = [s.replace('feature', '') for s in selected_features]
    l_int = list(map(int, l_replace))

    l_int.sort()
    l_string = list(map(str, l_int))
    selected_features_sorted = ['feature' + s for s in l_string]
    print(selected_features, selected_features_sorted)

    #Extract selected features in the correct order
    selected_features_df = features_df[selected_features_sorted]
    print("Feature Space Szie:", selected_features_df.shape)

    print(selected_features_df.columns)

    # Run the crip representation of the network features
    # features.dropna(inplace = True)
    myfeats = Feature(selected_features_df)
    # myfeats = Feature(features)
    print('Number of nodes:', myfeats.numNodes)
    print('nodes:', myfeats.nodes)
    myfeats.CrispRepresentation()

    np.random.seed(42)

    #Set Optimisation parameters -----------------------------------------------------------------------
    pDNA = pd.DataFrame(np.full([myfeats.numNodes,myfeats.numFeatures], 1),columns = myfeats.featureNames)
    hDNA = pd.DataFrame(np.full([myfeats.numNodes,myfeats.numFeatures], -1),columns = myfeats.featureNames)

    curr_results_output = []

    final_sim_data = None
    best_degree_dist = None
    opt_count = 0
    #---------------------------------------------------------------------------------------------------

    def blackbox(args):
        
        global MyEval, sim_data_df, myNetwork, node_degree_dist, opt_count, best_network, best_degree_dist, final_sim_data, optimized_pDNA, optimized_hDNA

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

        print('DNA Shapes:', pDNA.shape, hDNA.shape)
        myDNA = SocialDNA(myfeats,pDNA=pDNA,hDNA=hDNA)

        myNetwork = Network(myfeats, myDNA, NumEdge = edge_num, randomInterference=0.001)
        myNetwork.Formation(randomseed=50)

        #Evaluate the simulated network with the target network
        
        print('Num_nodes:', myNetwork.Graph.number_of_nodes())

        try:
            MyEval = NetworkEval(myNetwork.Graph,evalMetrics=["Density", "Modularity","Assortativity","Degree Distribution","Shortest Path Length Distribution","Clustering Coefficient Distribution"])

            MyEval.Similarity(G_target, numbins=100,zeroAlternative=10**(-7))
            sim_data_df = MyEval.SimilarityData
            Degree_Distribution = sim_data_df[sim_data_df['Metric'] == 'Degree Distribution']['Distance'].item()
            # Node_Degree = sim_data_df[sim_data_df['Metric'] == 'Node Degree']['Distance'].item()
            print("Degree_Distribution:", Degree_Distribution)
            # print("Sim data:", sim_data_df)

            if opt_count == 0:
                best_degree_dist = Degree_Distribution 
                final_sim_data = sim_data_df
                optimized_pDNA = pDNA
                optimized_hDNA = hDNA
                best_network = myNetwork
            
            if Degree_Distribution < best_degree_dist:
                best_degree_dist = Degree_Distribution 
                # node_degree_dist = Node_Degree 
                final_sim_data = sim_data_df
                optimized_pDNA = pDNA
                optimized_hDNA = hDNA
                best_network = myNetwork
                # optimized_pDNA.to_excel('Test/MyNetwork/optimized_pDNA.xlsx')
                # optimized_hDNA.to_excel('Test/MyNetwork/optimized_hDNA.xlsx')

            opt_count += 1

            print("curr opt loop:", opt_count)
            print("Best_dist:", best_degree_dist)
            
        except Exception as e:
            print("Error thrown:", e)
            try:
                print(Degree_Distribution)
            except:
                Degree_Distribution = 1

        return Degree_Distribution
        # return Node_Degree

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
                hp.uniform('H4', -1,1), hp.uniform('H5', -1,1),hp.uniform('H6', -1,1), hp.uniform('H7', -1,1), hp.uniform('H8', -1,1), hp.uniform('H9', -1,1), hp.uniform('H10', -1,1), hp.uniform('H11', -1,1)]

    h_space_curr = h_space[:select_features_num]

    # p_space = [hp.uniform('P1', -1,1),hp.uniform('P2', -1,1),hp.uniform('P3', -1,1),
    #             hp.uniform('P4', -1,1), hp.uniform('P5', -1,1),hp.uniform('P6', -1,1),hp.uniform('P7', -1,1),
    #             hp.uniform('P8', -1,1),hp.uniform('P9', -1,1),hp.uniform('P10', -1,1),
    #             hp.uniform('P11', -1,1), hp.uniform('P12', -1,1),hp.uniform('P13', -1,1),hp.uniform('P14', -1,1),
    #             hp.uniform('P15', -1,1),hp.uniform('P16', -1,1),hp.uniform('P17', -1,1),
    #             hp.uniform('P18', -1,1),hp.uniform('P19', -1,1),hp.uniform('P20', -1,1),
    #             hp.uniform('P21', -1,1), hp.uniform('P22', -1,1),hp.uniform('P23', -1,1),hp.uniform('P24', -1,1)]
    p_space = [hp.uniform('P1', -1,1),hp.uniform('P2', -1,1),hp.uniform('P3', -1,1),
                hp.uniform('P4', -1,1), hp.uniform('P5', -1,1),hp.uniform('P6', -1,1), hp.uniform('P7', -1,1), hp.uniform('P8', -1,1), hp.uniform('P9', -1,1), hp.uniform('P10', -1,1), hp.uniform('P11', -1,1)]

    p_space_curr = p_space[:select_features_num]

    space_curr = h_space_curr + p_space_curr
    # minimize the objective over the space
    best = fmin(blackbox, space_curr, algo=tpe.suggest, max_evals= 20, rstate=np.random.default_rng(seed = 42)) ##

    if i == 0:
        Overall_best_degree_dist = best_degree_dist
        num_features_final = select_features_num
        curr_results_output.append(select_features_num)
        curr_results_output.extend(final_sim_data['Distance'].to_list())
        curr_results_output.append(selected_features)
        print(curr_results_output)
        final_results_output.loc[i] = curr_results_output

    elif best_degree_dist < Overall_best_degree_dist:
        Overall_best_degree_dist = best_degree_dist
        num_features_final = select_features_num
        curr_results_output.append(select_features_num)
        curr_results_output.extend(final_sim_data['Distance'].to_list())
        curr_results_output.append(selected_features)
        print(curr_results_output)
        final_results_output.loc[i] = curr_results_output

    else:
        break

timespentonsim = time.time()-starttime

#Print out results
print(final_sim_data)
print(type(final_sim_data), final_sim_data.dtypes)
print("Time:",timespentonsim, "Score:", Overall_best_degree_dist, "Num Features:", num_features_final)

best_network.writeHistory(folder)
final_results_output.to_csv(result_output)