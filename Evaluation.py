import networkx as nx 
import pandas as pd 
import numpy as np 
import scipy 
import scipy.spatial
from community import community_louvain



class NetworkEval:
    def __init__(self,Graph,evalMetrics=["Density","Modularity","Assortativity","Degree Distribution","Shortest Path Length Distribution","Clustering Coefficient Distribution"]):
        self.evalMetrics = evalMetrics
        self.numNodes = len(Graph.nodes())
        resultslist =[]
        metricslist=[]
        for metric in evalMetrics:
            if metric=="Density":
                evalmetric = nx.density(Graph)
                resultslist.append(evalmetric)
                metricslist.append(metric)
            elif metric == "Modularity":
                #Changed Andreas 31/08
                try:
                    evalmetric = community_louvain.modularity(community_louvain.best_partition(Graph), Graph)
                    resultslist.append(evalmetric)
                    metricslist.append(metric)
                except:
                    print('ERRORED FOR MODULARITY')
                    resultslist.append(999)
                    metricslist.append(metric)
            elif metric =="Assortativity":
                evalmetric = nx.degree_assortativity_coefficient(Graph)
                resultslist.append(evalmetric)
                metricslist.append(metric)
            elif metric =="Degree Distribution":
                evalmetric = pd.DataFrame(nx.degree(Graph))
                evalmetric.columns=["node","Node Degree"]
                resultslist.append(evalmetric)
                metricslist.append(metric)
            elif metric =="Shortest Path Length Distribution":
                pathlist=[]
                [pathlist.extend([[node,nodej,nx.shortest_path_length(Graph,node,nodej)] if nx.has_path(Graph,node, nodej) else [node,nodej,self.numNodes] for nodej in range(node+1,self.numNodes)]) for node in range(self.numNodes-1)]
                evalmetric = pd.DataFrame(pathlist,columns=["node i","node j" ,"Shortest Path Length"])
                resultslist.append(evalmetric)
                metricslist.append(metric)

            elif metric =="Clustering Coefficient Distribution":
                evalmetric = pd.DataFrame()
                evalmetric["node"] = list(nx.clustering(Graph).keys())
                evalmetric["Clustering Coefficient"] = list(nx.clustering(Graph).values())
                resultslist.append(evalmetric)
                metricslist.append(metric)
            else:
                NotImplementedError 
    
        self.NetworkStatistics = dict(zip(evalMetrics,resultslist))

    def Similarity(self,targetGraph,numbins=100,zeroAlternative=10**(-5)):
        target = NetworkEval(targetGraph,evalMetrics = self.evalMetrics)

        #def SimilarityEval(self.numNodes,metricslist,NetSimEval,TargetEval):
        resultslist =[]
        for metric in self.evalMetrics:
            if metric in ["Density","Modularity"]:
                resultslist.append([metric,abs(self.NetworkStatistics[metric]-target.NetworkStatistics[metric])])
            elif metric == "Assortativity":
                resultslist.append([metric,abs(self.NetworkStatistics[metric]-target.NetworkStatistics[metric])/2])
            elif metric == "Degree Distribution":
                resultslist.append([metric,self.JSeval(self, self.NetworkStatistics[metric]["Node Degree"],target.NetworkStatistics[metric]["Node Degree"],metric,self.numNodes,valuetype="Discrete",numbins=numbins,zeroAlternative=zeroAlternative)])
            elif metric =="Shortest Path Length Distribution":
                resultslist.append([metric,self.JSeval(self, self.NetworkStatistics[metric]["Shortest Path Length"],target.NetworkStatistics[metric]["Shortest Path Length"],metric,self.numNodes,valuetype="Discrete",numbins=numbins,zeroAlternative=zeroAlternative)])
            elif metric == "Clustering Coefficient Distribution":
                resultslist.append([metric,self.JSeval(self, self.NetworkStatistics[metric]["Clustering Coefficient"],target.NetworkStatistics[metric]["Clustering Coefficient"],metric,self.numNodes,valuetype="Continuous",numbins=numbins,zeroAlternative=zeroAlternative)])
                
            else:
                NotImplementedError
        resultdata = pd.DataFrame(resultslist,columns=["Metric","Distance"])
        self.SimilarityData = resultdata

    @staticmethod
    def JSeval(self,sequence1,sequence2,metric,numNodes,valuetype="Continuous",numbins=100,zeroAlternative = 10**(-5)):
        minvalue = min(min(sequence1),min(sequence2))
        maxvalue = max(max(sequence1),max(sequence2))
        if minvalue ==maxvalue:
            return 0
        elif valuetype == "Discrete":
            if metric=="Shortest Path Length Distribution":
                possiblepath = set(sequence1+sequence2)
                try:
                    possiblepath.remove(self.numNodes)
                    bins = list(range(int(min(list(possiblepath)))-1,int(max(list(possiblepath))+1)))+[self.numNodes]
                except:
                    bins = list(range(int(min(list(possiblepath)))-1,int(max(list(possiblepath))+1)))
            else:
                bins = list(range(minvalue-1,maxvalue+1))
        
        else:
            #numbins = 100
            bins = [minvalue+i*(maxvalue-minvalue)/numbins for i in range(numbins)]

        num1 = np.array(pd.value_counts(pd.cut(sequence1,bins,include_lowest=True),sort=False).values)
        num2 = np.array(pd.value_counts(pd.cut(sequence2,bins,include_lowest=True),sort=False).values)

        freq1=num1/sum(num1)
        freq2=num2/sum(num2)
        
        adjustedFreq1 =  [x-zeroAlternative*(freq1==0).sum()/(len(num1)-(freq1==0).sum()) if x>0 else zeroAlternative for x in freq1]
        adjustedFreq2 =  [x-zeroAlternative*(freq2==0).sum()/(len(num1)-(freq2==0).sum()) if x>0 else zeroAlternative for x in freq2]

        JS = scipy.spatial.distance.jensenshannon(adjustedFreq1,adjustedFreq2)
    
        return JS








