import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def featureSelection(features_df = None):
    feature_ranking_variance = pd.DataFrame()
    feature_ranking_multi_collinearity = pd.DataFrame()
    feature_ranking = pd.DataFrame()
    
    feature_list = []
    variance_list = []
    multi_collinearity = []

    features = list(features_df.columns)
    num_features = len(features)
    
    feature_sim_data = pd.DataFrame()

    for i in range(num_features):
        current_feature_name = str(features[i])
        current_feature = list(features_df[current_feature_name])
        curFeature = list((current_feature - np.min(current_feature)) / (np.max(current_feature) - np.min(current_feature)))
        feature_sim_data['feature' + str(i)] = curFeature

    #drop columns which have null values after standardization
    before_drop = feature_sim_data.shape[1]
    feature_sim_data.dropna(axis = 1, inplace = True)
    after_drop = feature_sim_data.shape[1]
    shape_diff = before_drop - after_drop
    print("Num of dropped columns:", shape_diff)

    if shape_diff > 0:
        num_features = len(feature_sim_data.columns)
    sim_data_feature_names = list(feature_sim_data.columns)

    for i in range(num_features):
        current_feature_name = str(sim_data_feature_names[i])
        current_feature = list(feature_sim_data[current_feature_name])

        # For feature selection based on variance:
        curr_var = np.var(current_feature)

        # For feature selection based on multi_collinearity
        curr_multi_col = variance_inflation_factor(feature_sim_data.values, i)

        feature_list.append(current_feature_name)
        variance_list.append(curr_var)
        multi_collinearity.append(curr_multi_col)
        
    #Add feature names to each feature ranking dataframe
    feature_ranking_variance['Feature'] = feature_list
    feature_ranking_multi_collinearity['Feature'] = feature_list

    #Create each feature ranking and sort them by their respective selection method
    feature_ranking_variance['Variance'] = variance_list
    feature_ranking_variance['Variance_stand'] = list((feature_ranking_variance['Variance'] - np.min(feature_ranking_variance['Variance'])) / (np.max(feature_ranking_variance['Variance']) - np.min(feature_ranking_variance['Variance'])))
    feature_ranking_variance.sort_values(by = ['Variance_stand'], ascending = False, inplace = True)
    feature_ranking_variance.reset_index(drop = True, inplace = True)
    feature_ranking_variance['Feature_ranking_Va'] = 1
    feature_ranking_variance['Feature_ranking_Va'] = feature_ranking_variance['Feature_ranking_Va'].cumsum()
    print(feature_ranking_variance)

    feature_ranking_multi_collinearity['Multi_Collinearity'] = multi_collinearity
    feature_ranking_multi_collinearity['Multi_Collinearity_stand'] = list((feature_ranking_multi_collinearity['Multi_Collinearity'] - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])) / (np.max(feature_ranking_multi_collinearity['Multi_Collinearity']) - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])))
    feature_ranking_multi_collinearity['Multi_Collinearity_flipped'] = 1 - feature_ranking_multi_collinearity['Multi_Collinearity_stand'] 
    feature_ranking_multi_collinearity.sort_values(by = ['Multi_Collinearity_flipped'], ascending = False, inplace = True)
    feature_ranking_multi_collinearity.reset_index(drop = True, inplace = True)
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = 1
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = feature_ranking_multi_collinearity['Feature_ranking_MC'].cumsum()
    print(feature_ranking_multi_collinearity)

    #Create the final feature ranking df
    feature_ranking = pd.merge(feature_ranking_multi_collinearity, feature_ranking_variance, how = 'left', on = ['Feature'])

    #Create a ranking count to see overlap
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 1, 1, 0)
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 1, feature_ranking['1st_ranking_count'] + 1, feature_ranking['1st_ranking_count'])
    # feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 1, feature_ranking['1st_ranking_count'] + 1, feature_ranking['1st_ranking_count'])
    feature_ranking['1st_ranking_count'] = feature_ranking['1st_ranking_count']*3

    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 2, 1, 0)
    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 2, feature_ranking['2nd_ranking_count'] + 1, feature_ranking['2nd_ranking_count'])
    # feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 2, feature_ranking['2nd_ranking_count'] + 1, feature_ranking['2nd_ranking_count'])
    feature_ranking['2nd_ranking_count'] = feature_ranking['2nd_ranking_count']*2

    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 3, 1, 0)
    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 3, feature_ranking['3rd_ranking_count'] + 1, feature_ranking['3rd_ranking_count'])
    # feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 3, feature_ranking['3rd_ranking_count'] + 1, feature_ranking['3rd_ranking_count'])

    feature_ranking['Final_weight'] =  (feature_ranking['1st_ranking_count'] + feature_ranking['2nd_ranking_count'] + feature_ranking['3rd_ranking_count'])/3
    feature_ranking['Final_weight'] = np.where(feature_ranking['Final_weight'] < 1, 1, feature_ranking['Final_weight'])

    feature_ranking['Multi_Collinearity_flipped'].fillna(0, inplace = True)
    feature_ranking['Final_Filter'] = (feature_ranking['Variance_stand'] + feature_ranking['Multi_Collinearity_flipped']) * feature_ranking['Final_weight']
    feature_ranking.sort_values(by = ['Final_Filter'], ascending = False, inplace = True)
    feature_ranking.reset_index(drop = True, inplace = True)

    return feature_ranking, feature_sim_data