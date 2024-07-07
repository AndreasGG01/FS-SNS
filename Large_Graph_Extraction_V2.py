from Feature import *
from Network import * 
from FeatureExtraction import *
import time
import random
import matplotlib.pyplot as plt

def large_graph_extraction_v2(file_path, output_path, output_edge_list):
    colnames=['node1', 'node2'] 
    large_graph_df = pd.read_csv(file_path, names=colnames, header=None)
    large_graph_df.drop(index=large_graph_df.index[0], axis=0, inplace=True)
    print(large_graph_df.shape)

    large_graph_df.dropna(inplace = True)

    large_graph_df['node1'] = large_graph_df['node1'].astype(float) 
    large_graph_df['node2'] = large_graph_df['node2'].astype(float)

    print(large_graph_df.shape)

    unique_node1_list = large_graph_df['node1'].values.tolist()
    print(len(unique_node1_list))

    selection_list = ['node1', 'node2']
    selection = selection_list[0]
    selection_2 = selection_list[1]
    print(selection, selection_2)
    node = np.random.choice(unique_node1_list)
    node = float(node)

    final_df = pd.DataFrame(columns = ['node1', 'node2'])
    print(final_df)
    final_shape = final_df.shape[0]

    #Change the while loop value to change the number of nodes included in the graph
    while (final_shape <= 10000):
        curr_df = large_graph_df[large_graph_df[selection] == node]
        print('curr_Shape:',curr_df.shape)

        samples = random.randint(1, 5)

        if curr_df.empty:
            print('No More Connections')
            break
        elif curr_df.shape[0] < samples:
            samples = curr_df.shape[0]
            sampled_df = curr_df.sample(n = samples, axis = 0)
        else:
            sampled_df = curr_df.sample(n = samples, axis = 0)
        
        print('sampled_df', sampled_df.shape)

        curr_list = sampled_df[selection_2].values.tolist()
        node = random.choice(curr_list)

        selection, selection_2 = selection_2, selection

        final_df = pd.concat([final_df, sampled_df])
        final_shape = final_df.shape[0]
        print(final_shape)

    final_df.to_csv(output_edge_list)

    edge_list_df = final_df
    G_target = nx.from_pandas_edgelist(edge_list_df, 'node1', 'node2', create_using= nx.Graph())
    list_edges = list(G_target.edges(data=True))

    nx.write_graphml(G_target, output_path)
    # print('List_Edges:', list_edges)

    # Draw the networks in each subplot
    nx.draw_networkx(G_target, node_color='r', with_labels=False, node_size=10)

    plt.savefig('Test\\deezer_graph_extraction_1000.png')
    plt.show()

    return G_target