import numpy as np
import pandas as pd
import networkx as nx
import math
import copy 
import os




class Feature:

    def __init__(self,feature):
        print("USING NEW")
        ## feature is a dataframe, with columns of feature names, index from 0 to N-1 for N nodes.
        self.feature = feature
        self.numNodes = feature.shape[0]#len(feature)
        self.numFeatures = feature.shape[1]
        self.featureNames = feature.columns
        self.nodes = list(range(self.numNodes))

    def CrispRepresentation(self):
        ## feature: a dataframe, with its values ranging between 0 and 1
        ## featureDifferenceDict: a dictionary, featureDifferenceDict[0] means the feature difference dataframe between node 0 and other nodes.
        try:
            self.feature = (self.feature-self.feature.min(axis=0))/(self.feature.max(axis=0)-self.feature.min(axis=0)) 
        except:
            self.feature = self.feature*0
            
        featdiff = copy.deepcopy(self.feature)
        featdiff.columns = [name+" Difference" for name in featdiff.columns]
        self.featureDifferenceDict = dict(zip(self.nodes,[(featdiff - featdiff.loc[node]).abs() for node in self.nodes]))
        self.numFeatureDifferences = self.featureDifferenceDict[0].shape[1]
        self.featureNames = self.feature.columns

        self.NumFuzzyPdict = {}
        self.NumFuzzyHdict = {}
    
        self.FuzzyPparamsDict = {}
        self.FuzzyHparamsDict = {}

    def FuzzyRepresentation(self,NumFuzzyPdict,NumFuzzyHdict,minValue=None,maxValue=None,minDiffValue=None,maxDiffValue=None,FuzzyPparamsDict=None,FuzzyHparamsDict=None):
        ## NumFuzzyPlist=【"feat1":1，"feat2":2，"feat3":5，"feat4":3，"feat5":5】 
        ## NumFuzzyHlist= 【"featDiff1":1，"featDiff2":2，"featDiff3":5，"featDiff4":3，"featDiff5":5】
        ## FuzzyPparams = Dict[dataframes]
        self.NumFuzzyPdict =NumFuzzyPdict
        self.NumFuzzyHdict =NumFuzzyHdict
        self.FuzzyPparamsDict = FuzzyPparamsDict 
        self.FuzzyHparamsDict = FuzzyHparamsDict
        self.featureDifferenceDict = dict(zip(self.nodes,[(self.feature - self.feature.loc[node]).abs() for node in self.nodes]))
        
        if type(minValue)==type(None):
            minfeatvalues= self.feature.min(axis=0)
        else:
            minfeatvalues = minValue
        if type(maxValue)==type(None):
            maxfeatvalues = self.feature.max(axis=0)
        else:
            maxfeatvalues = maxValue
        if FuzzyPparamsDict ==None:
            ParamsDatalist=[]
            for featurename in self.featureNames:
                Fuzzydata=pd.DataFrame()
                if NumFuzzyPdict[featurename] ==1:
                    Fuzzydata["mu"] = [minfeatvalues[featurename]]
                    Fuzzydata["sigma"]=5
                else:
                    Fuzzydata["mu"] = np.linspace(minfeatvalues[featurename], maxfeatvalues[featurename],num=NumFuzzyPdict[featurename])
                    Fuzzydata["sigma"]=5
                ParamsDatalist.append(Fuzzydata)
            self.FuzzyPparamsDict = dict(zip(self.featureNames,ParamsDatalist))
        else:
            pass             

        if FuzzyHparamsDict ==None:
            ParamsDatalist=[]
            for featurename in self.featureNames:
                if (type(minDiffValue)==type(None)):
                    
                    minfeatDiffs= np.min([self.featureDifferenceDict[node][featurename].min(axis=0) for node in range(self.numNodes)])
                  #  print(minfeatDiffs)
                else:
                    minfeatDiffs = minDiffValue[featurename]
                if (type(maxDiffValue)==type(None)):
                    maxfeatDiffs = np.max([self.featureDifferenceDict[node][featurename].max(axis=0) for node in range(self.numNodes)])
                else:
                    maxfeatDiffs = maxDiffValue[featurename]

                Fuzzydata=pd.DataFrame()         
                if NumFuzzyHdict[featurename] ==1:
                    Fuzzydata["mu"] = [minfeatDiffs]
                    Fuzzydata["sigma"]=5
                else:
                    Fuzzydata["mu"] = np.linspace(minfeatDiffs, maxfeatDiffs,num=NumFuzzyHdict[featurename])
                    Fuzzydata["sigma"]=5
                ParamsDatalist.append(Fuzzydata)
            self.FuzzyHparamsDict= dict(zip(self.featureNames,ParamsDatalist))
        else:
            pass             

        NewFeatureData,featcolumns,featDiffcolumns=[],[],[]
        featureDiffNew = [[] for i in range(self.numNodes)] 

        for featurename in self.featureNames:
            featureNew = [list(np.exp(-(self.feature[featurename]-self.FuzzyPparamsDict[featurename].loc[i,"mu"])**2/(2*self.FuzzyPparamsDict[featurename].loc[i,"sigma"]**2)).values) for i in range(self.NumFuzzyPdict[featurename])]
            NewFeatureData.extend(featureNew)
            featcolumns.extend([featurename+"-Fuzzy"+str(i) for i in range(self.NumFuzzyPdict[featurename])])
            featDiffcolumns.extend([featurename+" Difference-Fuzzy"+str(i) for i in range(self.NumFuzzyHdict[featurename])])
            featureDiffNew=dict(zip(list(range(self.numNodes)),[featureDiffNew[node]+[list(np.exp(-(self.featureDifferenceDict[node][featurename]-self.FuzzyHparamsDict[featurename].loc[i,"mu"])**2/(2*self.FuzzyHparamsDict[featurename].loc[i,"sigma"]**2)).values) for i in range(self.NumFuzzyHdict[featurename])] for node in range(self.numNodes) ]))

        featureDiffNew = dict(zip(list(range(self.numNodes)),[pd.DataFrame(featureDiffNew[node],index=featDiffcolumns).T for node in range(self.numNodes)]))
        NewFeatureData = pd.DataFrame(NewFeatureData,index=featcolumns).T

        self.feature = NewFeatureData
        self.featureDifferenceDict = featureDiffNew#NewFeatureData
        self.numFeatures = self.feature.shape[1]
        self.numFeatureDifferences = self.featureDifferenceDict[0].shape[1]
        self.featureNames = self.feature.columns
        

    def FeatureAppend(self,AnotherFeature):
        self.feature = pd.concat([self.feature,AnotherFeature.feature],axis=1)
        self.numFeatures = self.feature.shape[1]
        self.featureNames = self.feature.columns
        self.featureDifferenceDict = dict(zip(list(range(self.numNodes)),[pd.concat([self.featureDifferenceDict[node],AnotherFeature.featureDifferenceDict[node]],axis=1) for node in range(self.numNodes)]))
        self.numFeatureDifferences = self.featureDifferenceDict[0].shape[1]
        
        if (type(self.NumFuzzyPdict)==type(None)) or (type(self.NumFuzzyHdict)==type(None)):
                
                self.NumFuzzyPdict=None
                self.NumFuzzyHdict=None
                self.FuzzyPparamsDict = None
                self.FuzzyHparamsDict = None
        else:
            self.NumFuzzyPdict = self.NumFuzzyPdict.update(AnotherFeature.NumFuzzyPdict)
            self.NumFuzzyHdict = self.NumFuzzyHdict.update(AnotherFeature.NumFuzzyHdict)
    
            self.FuzzyPparamsDict = self.FuzzyPparamsDict.update(AnotherFeature.FuzzyPparamsDict)
            self.FuzzyHparamsDict = self.FuzzyHparamsDict.update(AnotherFeature.FuzzyHparamsDict)
                                                             
    def FeatureUpdate(self,AnotherFeature,ReplaceFeatures=None):
        if type(ReplaceFeatures)==type(None):
            pass #self.FeatureAppend(AnotherFeature)
        else:                         
            newfeat = copy.deepcopy(self.feature)
            for featurename in self.feature.columns:
                for replacedfeat in ReplaceFeatures: 
                    if (featurename.find(replacedfeat)!=-1):
                        newfeat = newfeat.drop(featurename, axis=1)  
            self.feature = newfeat
            for featureDiffname in self.featureDifferenceDict[0].columns:
                [[self.featureDifferenceDict.update({node:self.featureDifferenceDict[node].drop(featureDiffname, axis=1)}) for replacedfeat in ReplaceFeatures if (featureDiffname.find(replacedfeat)!=-1) ] for node in self.nodes]
            
            if (type(self.NumFuzzyPdict)==type(None)) or (type(self.NumFuzzyHdict)==type(None)):
                pass 
            else:
                [self.NumFuzzyPdict.pop(replacedfeat, None) for replacedfeat in ReplaceFeatures]
                [self.NumFuzzyHdict.pop(replacedfeat, None) for replacedfeat in ReplaceFeatures]
                [self.FuzzyPparamsDict.pop(replacedfeat, None) for replacedfeat in ReplaceFeatures]
                [self.FuzzyHparamsDict.pop(replacedfeat, None) for replacedfeat in ReplaceFeatures]
            
        self.FeatureAppend(AnotherFeature)
            
            
 

                                                             

