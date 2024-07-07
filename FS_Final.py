import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from Filter_Methods.lap_score import *
import copy

def FS_Lap(features_df = None):
    feature_ranking_variance = pd.DataFrame()
    feature_ranking_multi_collinearity = pd.DataFrame()
    feature_ranking_lap_score = pd.DataFrame()
    feature_ranking_correlation = pd.DataFrame()
    feature_ranking_mutual_info = pd.DataFrame()
    feature_ranking = pd.DataFrame()

    variance_list = []
    multi_collinearity = []
    lap_score_list = []
    mutual_information = []

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
    feature_list = sim_data_feature_names

    #Check to see which features are discrete and which are continuous
    feature_discrete_check = pd.DataFrame()
    feature_discrete_check['Feature'] = sim_data_feature_names
    feature_discrete_check['Discrete'] = False

    for feature_dc in sim_data_feature_names:
        current_feature = list(feature_sim_data[feature_dc])
        if all(item in {0, 1} for item in current_feature):
            print('Value is discrete')
            feature_discrete_check['Discrete'] = np.where(feature_discrete_check['Feature'] == feature_dc, True, feature_discrete_check['Discrete'])
    

    # #loop to calculate variance, multi collinearity and mutual info
    # for i in range(num_features):
    #     current_feature_name = str(sim_data_feature_names[i])
    #     current_feature = list(feature_sim_data[current_feature_name])

    #     discrete = feature_discrete_check[feature_discrete_check['Feature'] == current_feature_name]['Discrete'].item()
    #     # For feature selection based on variance:
    #     curr_var = np.var(current_feature)

    #     # For feature selection based on multi_collinearity
    #     curr_multi_col = variance_inflation_factor(feature_sim_data.values, i)
        
    #     #For feature selection based on mutual information
    #     mi_list_for_curr_feature = []
    #     second_loop_features = copy.deepcopy(sim_data_feature_names)
    #     second_loop_features.remove(second_loop_features[i])
    #     for feature_mi in sim_data_feature_names:
    #         curr_feature_discrete = feature_discrete_check[feature_discrete_check['Feature'] == feature_mi]['Discrete'].item()
    #         # print(feature_mi, curr_feature_discrete, current_feature_name, discrete)
    #         if curr_feature_discrete:
    #             mi = mutual_info_classif(feature_sim_data[current_feature_name].to_frame(), feature_sim_data[feature_mi], discrete_features=[discrete])
    #         else:
    #             mi = mutual_info_regression(feature_sim_data[current_feature_name].to_frame(), feature_sim_data[feature_mi], discrete_features=[discrete])
            
    #         mi_list_for_curr_feature.append(mi)
    #         curr_feature_mi_summed = sum(mi_list_for_curr_feature)

    #     #Append all the calculated values
        # feature_list.append(current_feature_name)
    #     variance_list.append(curr_var)
    #     multi_collinearity.append(curr_multi_col)
    #     mutual_information.append(curr_feature_mi_summed[0])

    # print('mutula info print:',mutual_information) 

    #Calculate the lap score
    feature_np_array = feature_sim_data.to_numpy(copy=True)
    # print('Feature_array:', feature_np_array)
    lap_result = lap_score(feature_np_array)
    # print("Lap lacian score:",lap_result)

    # #Calculate Correlation
    # correlation_matrix = feature_sim_data.corr()
    # print(correlation_matrix)
    # correlation_matrix = correlation_matrix.abs()
    # feature_ranking_correlation['Feature'] = feature_list
    # feature_ranking_correlation['Correlation_Sum'] = 0
    # for feature in sim_data_feature_names:
    #     curr_correlation_sum = correlation_matrix[feature].sum() - 1
    #     feature_ranking_correlation['Correlation_Sum'] = np.where(feature_ranking_correlation['Feature'] == feature, curr_correlation_sum, feature_ranking_correlation['Correlation_Sum'])

    

    #Compare feature list with sim_data_feature_names
    # print('feature_list', feature_list)
    # print('feature_sim_names:', sim_data_feature_names)

    #Add feature names to each feature ranking dataframe
    # feature_ranking_variance['Feature'] = feature_list
    # feature_ranking_multi_collinearity['Feature'] = feature_list
    feature_ranking_lap_score['Feature'] = feature_list
    # feature_ranking_mutual_info['Feature'] = feature_list

    #Create each feature ranking and sort them by their respecfeature_listtive selection method
    # feature_ranking_variance['Variance'] = variance_list
    # feature_ranking_variance['Variance'].replace([np.inf, -np.inf], 0, inplace=True)
    # feature_ranking_variance['Variance_stand'] = list((feature_ranking_variance['Variance'] - np.min(feature_ranking_variance['Variance'])) / (np.max(feature_ranking_variance['Variance']) - np.min(feature_ranking_variance['Variance'])))
    # feature_ranking_variance.sort_values(by = ['Variance_stand'], ascending = False, inplace = True)
    # feature_ranking_variance.reset_index(drop = True, inplace = True)
    # feature_ranking_variance['Feature_ranking_Va'] = 1
    # feature_ranking_variance['Feature_ranking_Va'] = feature_ranking_variance['Feature_ranking_Va'].cumsum()
    # print(feature_ranking_variance)

    # feature_ranking_multi_collinearity['Multi_Collinearity'] = multi_collinearity
    # feature_ranking_multi_collinearity['Multi_Collinearity'].replace([np.inf, -np.inf], 0, inplace=True)
    # feature_ranking_multi_collinearity['Multi_Collinearity_stand'] = list((feature_ranking_multi_collinearity['Multi_Collinearity'] - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])) / (np.max(feature_ranking_multi_collinearity['Multi_Collinearity']) - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])))
    # feature_ranking_multi_collinearity['Multi_Collinearity_flipped'] = 1 - feature_ranking_multi_collinearity['Multi_Collinearity_stand'] 
    # feature_ranking_multi_collinearity.sort_values(by = ['Multi_Collinearity_flipped'], ascending = False, inplace = True)
    # feature_ranking_multi_collinearity.reset_index(drop = True, inplace = True)
    # feature_ranking_multi_collinearity['Feature_ranking_MC'] = 1
    # feature_ranking_multi_collinearity['Feature_ranking_MC'] = feature_ranking_multi_collinearity['Feature_ranking_MC'].cumsum()
    # print(feature_ranking_multi_collinearity)

    feature_ranking_lap_score['Lap_Score'] = lap_result
    feature_ranking_lap_score['Lap_Score_stand'] = list((feature_ranking_lap_score['Lap_Score'] - np.min(feature_ranking_lap_score['Lap_Score'])) / (np.max(feature_ranking_lap_score['Lap_Score']) - np.min(feature_ranking_lap_score['Lap_Score'])))
    feature_ranking_lap_score['Lap_Score_flipped'] = 1 - feature_ranking_lap_score['Lap_Score_stand'] 
    feature_ranking_lap_score.sort_values(by = ['Lap_Score_flipped'], ascending = False, inplace = True)
    feature_ranking_lap_score.reset_index(drop = True, inplace = True)
    feature_ranking_lap_score['Feature_ranking_LP'] = 1
    feature_ranking_lap_score['Feature_ranking_LP'] = feature_ranking_lap_score['Feature_ranking_LP'].cumsum()
    print(feature_ranking_lap_score)

    # feature_ranking_mutual_info['Mutual_Information'] = mutual_information
    # feature_ranking_mutual_info['Mutual_Information_Stand'] = list((feature_ranking_mutual_info['Mutual_Information'] - np.min(feature_ranking_mutual_info['Mutual_Information'])) / (np.max(feature_ranking_mutual_info['Mutual_Information']) - np.min(feature_ranking_mutual_info['Mutual_Information'])))
    # feature_ranking_mutual_info.sort_values(by = ['Mutual_Information_Stand'], ascending = False, inplace = True)
    # feature_ranking_mutual_info.reset_index(drop = True, inplace = True)
    # feature_ranking_mutual_info['Feature_ranking_MI'] = 1
    # feature_ranking_mutual_info['Feature_ranking_MI'] = feature_ranking_mutual_info['Feature_ranking_MI'].cumsum()
    # print(feature_ranking_mutual_info)

    #Create the final feature ranking df
    # feature_ranking = pd.merge(feature_ranking_multi_collinearity, feature_ranking_variance, how = 'left', on = ['Feature'])
    # feature_ranking = pd.merge(feature_ranking, feature_ranking_lap_score, how = 'left', on = ['Feature'])
    # feature_ranking = pd.merge(feature_ranking, feature_ranking_correlation, how = 'left', on = ['Feature'])
    # feature_ranking = pd.merge(feature_ranking, feature_ranking_mutual_info, how = 'left', on = ['Feature'])

    feature_ranking = feature_ranking_lap_score

    #Create a ranking count to see overlap
    # feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 1, 1, 0)
    # feature_ranking['1st_ranking_count'] = feature_ranking['1st_ranking_count']*3

    # feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 2, 1, 0)
    # feature_ranking['2nd_ranking_count'] = feature_ranking['2nd_ranking_count']*2

    # feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 3, 1, 0)

    # feature_ranking['Final_weight'] =  (feature_ranking['1st_ranking_count'] + feature_ranking['2nd_ranking_count'] + feature_ranking['3rd_ranking_count'])/3
    # feature_ranking['Final_weight'] = np.where(feature_ranking['Final_weight'] < 1, 1, feature_ranking['Final_weight'])

    # feature_ranking['Final_Filter'] = (feature_ranking['Variance_stand'] + feature_ranking['Multi_Collinearity_flipped'] + feature_ranking['Lap_Score_flipped'] + feature_ranking['Correlation_Flipped'] + feature_ranking['Mutual_Information_Stand']) * feature_ranking['Final_weight']
    feature_ranking['Final_Filter'] = feature_ranking['Lap_Score_flipped']

    feature_ranking.sort_values(by = ['Final_Filter'], ascending = False, inplace = True)
    feature_ranking.reset_index(drop = True, inplace = True)

    return feature_ranking, feature_sim_data

def FS_Var(features_df = None):
    feature_ranking_variance = pd.DataFrame()
    feature_ranking_multi_collinearity = pd.DataFrame()
    feature_ranking_lap_score = pd.DataFrame()
    feature_ranking_correlation = pd.DataFrame()
    feature_ranking_mutual_info = pd.DataFrame()
    feature_ranking = pd.DataFrame()

    variance_list = []
    multi_collinearity = []
    lap_score_list = []
    mutual_information = []

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
    feature_list = sim_data_feature_names

    #Check to see which features are discrete and which are continuous
    feature_discrete_check = pd.DataFrame()
    feature_discrete_check['Feature'] = sim_data_feature_names
    feature_discrete_check['Discrete'] = False

    for feature_dc in sim_data_feature_names:
        current_feature = list(feature_sim_data[feature_dc])
        if all(item in {0, 1} for item in current_feature):
            print('Value is discrete')
            feature_discrete_check['Discrete'] = np.where(feature_discrete_check['Feature'] == feature_dc, True, feature_discrete_check['Discrete'])
    

    #loop to calculate variance, multi collinearity and mutual info
    for i in range(num_features):
        current_feature_name = str(sim_data_feature_names[i])
        current_feature = list(feature_sim_data[current_feature_name])

        discrete = feature_discrete_check[feature_discrete_check['Feature'] == current_feature_name]['Discrete'].item()
        # For feature selection based on variance:
        curr_var = np.var(current_feature)

        variance_list.append(curr_var)

    #Add feature names to each feature ranking dataframe
    feature_ranking_variance['Feature'] = feature_list

    #Create each feature ranking and sort them by their respecfeature_listtive selection method
    feature_ranking_variance['Variance'] = variance_list
    feature_ranking_variance['Variance'].replace([np.inf, -np.inf], 0, inplace=True)
    feature_ranking_variance['Variance_stand'] = list((feature_ranking_variance['Variance'] - np.min(feature_ranking_variance['Variance'])) / (np.max(feature_ranking_variance['Variance']) - np.min(feature_ranking_variance['Variance'])))
    feature_ranking_variance.sort_values(by = ['Variance_stand'], ascending = False, inplace = True)
    feature_ranking_variance.reset_index(drop = True, inplace = True)
    feature_ranking_variance['Feature_ranking_Va'] = 1
    feature_ranking_variance['Feature_ranking_Va'] = feature_ranking_variance['Feature_ranking_Va'].cumsum()
    print(feature_ranking_variance)

    feature_ranking = feature_ranking_variance

    feature_ranking['Final_Filter'] = feature_ranking['Variance_stand']

    feature_ranking.sort_values(by = ['Final_Filter'], ascending = False, inplace = True)
    feature_ranking.reset_index(drop = True, inplace = True)

    return feature_ranking, feature_sim_data

def FS_MultiCol(features_df = None):
    feature_ranking_variance = pd.DataFrame()
    feature_ranking_multi_collinearity = pd.DataFrame()
    feature_ranking_lap_score = pd.DataFrame()
    feature_ranking_correlation = pd.DataFrame()
    feature_ranking_mutual_info = pd.DataFrame()
    feature_ranking = pd.DataFrame()

    variance_list = []
    multi_collinearity = []
    lap_score_list = []
    mutual_information = []

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
    feature_list = sim_data_feature_names

    #Check to see which features are discrete and which are continuous
    feature_discrete_check = pd.DataFrame()
    feature_discrete_check['Feature'] = sim_data_feature_names
    feature_discrete_check['Discrete'] = False

    for feature_dc in sim_data_feature_names:
        current_feature = list(feature_sim_data[feature_dc])
        if all(item in {0, 1} for item in current_feature):
            print('Value is discrete')
            feature_discrete_check['Discrete'] = np.where(feature_discrete_check['Feature'] == feature_dc, True, feature_discrete_check['Discrete'])
    

    # #loop to calculate variance, multi collinearity and mutual info
    for i in range(num_features):
        current_feature_name = str(sim_data_feature_names[i])
        current_feature = list(feature_sim_data[current_feature_name])

        discrete = feature_discrete_check[feature_discrete_check['Feature'] == current_feature_name]['Discrete'].item()

        # For feature selection based on multi_collinearity
        curr_multi_col = variance_inflation_factor(feature_sim_data.values, i)

        #Append all the calculated values
        multi_collinearity.append(curr_multi_col)

    # Compare feature list with sim_data_feature_names
    print('feature_list', feature_list)
    print('feature_sim_names:', sim_data_feature_names)

    #Add feature names to each feature ranking dataframe
    # feature_ranking_variance['Feature'] = feature_list
    feature_ranking_multi_collinearity['Feature'] = feature_list
    # feature_ranking_lap_score['Feature'] = feature_list
    # feature_ranking_mutual_info['Feature'] = feature_list

    #Create each feature ranking and sort them by their respecfeature_listtive selection method

    feature_ranking_multi_collinearity['Multi_Collinearity'] = multi_collinearity
    feature_ranking_multi_collinearity['Multi_Collinearity'].replace([np.inf, -np.inf], 0, inplace=True)
    feature_ranking_multi_collinearity['Multi_Collinearity_stand'] = list((feature_ranking_multi_collinearity['Multi_Collinearity'] - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])) / (np.max(feature_ranking_multi_collinearity['Multi_Collinearity']) - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])))
    feature_ranking_multi_collinearity['Multi_Collinearity_flipped'] = 1 - feature_ranking_multi_collinearity['Multi_Collinearity_stand'] 
    feature_ranking_multi_collinearity.sort_values(by = ['Multi_Collinearity_flipped'], ascending = False, inplace = True)
    feature_ranking_multi_collinearity.reset_index(drop = True, inplace = True)
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = 1
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = feature_ranking_multi_collinearity['Feature_ranking_MC'].cumsum()
    print(feature_ranking_multi_collinearity)

    feature_ranking = feature_ranking_multi_collinearity

    feature_ranking['Final_Filter'] = feature_ranking['Multi_Collinearity_flipped']

    feature_ranking.sort_values(by = ['Final_Filter'], ascending = False, inplace = True)
    feature_ranking.reset_index(drop = True, inplace = True)

    return feature_ranking, feature_sim_data

def FS_MutualInfo(features_df = None):
    feature_ranking_variance = pd.DataFrame()
    feature_ranking_multi_collinearity = pd.DataFrame()
    feature_ranking_lap_score = pd.DataFrame()
    feature_ranking_correlation = pd.DataFrame()
    feature_ranking_mutual_info = pd.DataFrame()
    feature_ranking = pd.DataFrame()

    variance_list = []
    multi_collinearity = []
    lap_score_list = []
    mutual_information = []

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
    feature_list = sim_data_feature_names

    #Check to see which features are discrete and which are continuous
    feature_discrete_check = pd.DataFrame()
    feature_discrete_check['Feature'] = sim_data_feature_names
    feature_discrete_check['Discrete'] = False

    for feature_dc in sim_data_feature_names:
        current_feature = list(feature_sim_data[feature_dc])
        if all(item in {0, 1} for item in current_feature):
            print('Value is discrete')
            feature_discrete_check['Discrete'] = np.where(feature_discrete_check['Feature'] == feature_dc, True, feature_discrete_check['Discrete'])
    

    #loop to calculate mutual info
    for i in range(num_features):
        current_feature_name = str(sim_data_feature_names[i])
        current_feature = list(feature_sim_data[current_feature_name])

        discrete = feature_discrete_check[feature_discrete_check['Feature'] == current_feature_name]['Discrete'].item()
        
        #For feature selection based on mutual information
        mi_list_for_curr_feature = []
        second_loop_features = copy.deepcopy(sim_data_feature_names)
        second_loop_features.remove(second_loop_features[i])
        for feature_mi in sim_data_feature_names:
            curr_feature_discrete = feature_discrete_check[feature_discrete_check['Feature'] == feature_mi]['Discrete'].item()
            # print(feature_mi, curr_feature_discrete, current_feature_name, discrete)
            if curr_feature_discrete:
                mi = mutual_info_classif(feature_sim_data[current_feature_name].to_frame(), feature_sim_data[feature_mi], discrete_features=[discrete])
            else:
                mi = mutual_info_regression(feature_sim_data[current_feature_name].to_frame(), feature_sim_data[feature_mi], discrete_features=[discrete])
            
            mi_list_for_curr_feature.append(mi)
            curr_feature_mi_summed = sum(mi_list_for_curr_feature)

        #Append all the calculated values
        mutual_information.append(curr_feature_mi_summed[0])

    print('mutula info print:',mutual_information)     

    # Compare feature list with sim_data_feature_names
    print('feature_list', feature_list)
    print('feature_sim_names:', sim_data_feature_names)

    #Add feature names to each feature ranking dataframe
    # feature_ranking_variance['Feature'] = feature_list
    # feature_ranking_multi_collinearity['Feature'] = feature_list
    # feature_ranking_lap_score['Feature'] = feature_list
    feature_ranking_mutual_info['Feature'] = feature_list

    #Create each feature ranking and sort them by their respecfeature_listtive selection method
    feature_ranking_mutual_info['Mutual_Information'] = mutual_information
    feature_ranking_mutual_info['Mutual_Information_Stand'] = list((feature_ranking_mutual_info['Mutual_Information'] - np.min(feature_ranking_mutual_info['Mutual_Information'])) / (np.max(feature_ranking_mutual_info['Mutual_Information']) - np.min(feature_ranking_mutual_info['Mutual_Information'])))
    feature_ranking_mutual_info.sort_values(by = ['Mutual_Information_Stand'], ascending = False, inplace = True)
    feature_ranking_mutual_info.reset_index(drop = True, inplace = True)
    feature_ranking_mutual_info['Feature_ranking_MI'] = 1
    feature_ranking_mutual_info['Feature_ranking_MI'] = feature_ranking_mutual_info['Feature_ranking_MI'].cumsum()
    print(feature_ranking_mutual_info)

    feature_ranking = feature_ranking_mutual_info

    feature_ranking['Final_Filter'] = feature_ranking['Mutual_Information_Stand']

    feature_ranking.sort_values(by = ['Final_Filter'], ascending = False, inplace = True)
    feature_ranking.reset_index(drop = True, inplace = True)

    return feature_ranking, feature_sim_data

def FS_All_Combined(features_df = None):
    feature_ranking_variance = pd.DataFrame()
    feature_ranking_multi_collinearity = pd.DataFrame()
    feature_ranking_lap_score = pd.DataFrame()
    feature_ranking_correlation = pd.DataFrame()
    feature_ranking_mutual_info = pd.DataFrame()
    feature_ranking = pd.DataFrame()

    variance_list = []
    multi_collinearity = []
    lap_score_list = []
    mutual_information = []

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
    feature_list = sim_data_feature_names

    #Check to see which features are discrete and which are continuous
    feature_discrete_check = pd.DataFrame()
    feature_discrete_check['Feature'] = sim_data_feature_names
    feature_discrete_check['Discrete'] = False

    for feature_dc in sim_data_feature_names:
        current_feature = list(feature_sim_data[feature_dc])
        if all(item in {0, 1} for item in current_feature):
            print('Value is discrete')
            feature_discrete_check['Discrete'] = np.where(feature_discrete_check['Feature'] == feature_dc, True, feature_discrete_check['Discrete'])
    

    #loop to calculate variance, multi collinearity and mutual info
    for i in range(num_features):
        current_feature_name = str(sim_data_feature_names[i])
        current_feature = list(feature_sim_data[current_feature_name])

        discrete = feature_discrete_check[feature_discrete_check['Feature'] == current_feature_name]['Discrete'].item()
        # For feature selection based on variance:
        curr_var = np.var(current_feature)

        # For feature selection based on multi_collinearity
        curr_multi_col = variance_inflation_factor(feature_sim_data.values, i)
        
        #For feature selection based on mutual information
        mi_list_for_curr_feature = []
        second_loop_features = copy.deepcopy(sim_data_feature_names)
        second_loop_features.remove(second_loop_features[i])
        for feature_mi in sim_data_feature_names:
            curr_feature_discrete = feature_discrete_check[feature_discrete_check['Feature'] == feature_mi]['Discrete'].item()
            # print(feature_mi, curr_feature_discrete, current_feature_name, discrete)
            if curr_feature_discrete:
                mi = mutual_info_classif(feature_sim_data[current_feature_name].to_frame(), feature_sim_data[feature_mi], discrete_features=[discrete])
            else:
                mi = mutual_info_regression(feature_sim_data[current_feature_name].to_frame(), feature_sim_data[feature_mi], discrete_features=[discrete])
            
            mi_list_for_curr_feature.append(mi)
            curr_feature_mi_summed = sum(mi_list_for_curr_feature)

        #Append all the calculated values
        variance_list.append(curr_var)
        multi_collinearity.append(curr_multi_col)
        mutual_information.append(curr_feature_mi_summed[0])

    print('mutula info print:',mutual_information) 

    #Calculate the lap score
    feature_np_array = feature_sim_data.to_numpy(copy=True)
    # print('Feature_array:', feature_np_array)
    lap_result = lap_score(feature_np_array)
    # print("Lap lacian score:",lap_result)    

    #Add feature names to each feature ranking dataframe
    feature_ranking_variance['Feature'] = feature_list
    feature_ranking_multi_collinearity['Feature'] = feature_list
    feature_ranking_lap_score['Feature'] = feature_list
    feature_ranking_mutual_info['Feature'] = feature_list

    #Create each feature ranking and sort them by their respecfeature_listtive selection method
    feature_ranking_variance['Variance'] = variance_list
    feature_ranking_variance['Variance'].replace([np.inf, -np.inf], 0, inplace=True)
    feature_ranking_variance['Variance_stand'] = list((feature_ranking_variance['Variance'] - np.min(feature_ranking_variance['Variance'])) / (np.max(feature_ranking_variance['Variance']) - np.min(feature_ranking_variance['Variance'])))
    feature_ranking_variance.sort_values(by = ['Variance_stand'], ascending = False, inplace = True)
    feature_ranking_variance.reset_index(drop = True, inplace = True)
    feature_ranking_variance['Feature_ranking_Va'] = 1
    feature_ranking_variance['Feature_ranking_Va'] = feature_ranking_variance['Feature_ranking_Va'].cumsum()
    print(feature_ranking_variance)

    feature_ranking_multi_collinearity['Multi_Collinearity'] = multi_collinearity
    feature_ranking_multi_collinearity['Multi_Collinearity'].replace([np.inf, -np.inf], 0, inplace=True)
    feature_ranking_multi_collinearity['Multi_Collinearity_stand'] = list((feature_ranking_multi_collinearity['Multi_Collinearity'] - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])) / (np.max(feature_ranking_multi_collinearity['Multi_Collinearity']) - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])))
    feature_ranking_multi_collinearity['Multi_Collinearity_flipped'] = 1 - feature_ranking_multi_collinearity['Multi_Collinearity_stand'] 
    feature_ranking_multi_collinearity.sort_values(by = ['Multi_Collinearity_flipped'], ascending = False, inplace = True)
    feature_ranking_multi_collinearity.reset_index(drop = True, inplace = True)
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = 1
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = feature_ranking_multi_collinearity['Feature_ranking_MC'].cumsum()
    print(feature_ranking_multi_collinearity)

    feature_ranking_lap_score['Lap_Score'] = lap_result
    feature_ranking_lap_score['Lap_Score_stand'] = list((feature_ranking_lap_score['Lap_Score'] - np.min(feature_ranking_lap_score['Lap_Score'])) / (np.max(feature_ranking_lap_score['Lap_Score']) - np.min(feature_ranking_lap_score['Lap_Score'])))
    feature_ranking_lap_score['Lap_Score_flipped'] = 1 - feature_ranking_lap_score['Lap_Score_stand'] 
    feature_ranking_lap_score.sort_values(by = ['Lap_Score_flipped'], ascending = False, inplace = True)
    feature_ranking_lap_score.reset_index(drop = True, inplace = True)
    feature_ranking_lap_score['Feature_ranking_LP'] = 1
    feature_ranking_lap_score['Feature_ranking_LP'] = feature_ranking_lap_score['Feature_ranking_LP'].cumsum()
    print(feature_ranking_lap_score)

    feature_ranking_mutual_info['Mutual_Information'] = mutual_information
    feature_ranking_mutual_info['Mutual_Information_Stand'] = list((feature_ranking_mutual_info['Mutual_Information'] - np.min(feature_ranking_mutual_info['Mutual_Information'])) / (np.max(feature_ranking_mutual_info['Mutual_Information']) - np.min(feature_ranking_mutual_info['Mutual_Information'])))
    feature_ranking_mutual_info.sort_values(by = ['Mutual_Information_Stand'], ascending = False, inplace = True)
    feature_ranking_mutual_info.reset_index(drop = True, inplace = True)
    feature_ranking_mutual_info['Feature_ranking_MI'] = 1
    feature_ranking_mutual_info['Feature_ranking_MI'] = feature_ranking_mutual_info['Feature_ranking_MI'].cumsum()
    print(feature_ranking_mutual_info)

    # Create the final feature ranking df
    feature_ranking = pd.merge(feature_ranking_multi_collinearity, feature_ranking_variance, how = 'left', on = ['Feature'])
    feature_ranking = pd.merge(feature_ranking, feature_ranking_lap_score, how = 'left', on = ['Feature'])
    feature_ranking = pd.merge(feature_ranking, feature_ranking_mutual_info, how = 'left', on = ['Feature'])

    #Create a ranking count to see overlap
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 1, 1, 0)
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 1, feature_ranking['1st_ranking_count'] + 1, feature_ranking['1st_ranking_count'])
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 1, feature_ranking['1st_ranking_count'] + 1, feature_ranking['1st_ranking_count'])
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_MI'] == 1, feature_ranking['1st_ranking_count'] + 1, feature_ranking['1st_ranking_count'])

    feature_ranking['1st_ranking_count'] = feature_ranking['1st_ranking_count']*3

    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 2, 1, 0)
    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 2, feature_ranking['2nd_ranking_count'] + 1, feature_ranking['2nd_ranking_count'])
    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 2, feature_ranking['2nd_ranking_count'] + 1, feature_ranking['2nd_ranking_count'])
    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MI'] == 2, feature_ranking['2nd_ranking_count'] + 1, feature_ranking['2nd_ranking_count'])
    feature_ranking['2nd_ranking_count'] = feature_ranking['2nd_ranking_count']*2

    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 3, 1, 0)
    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 3, feature_ranking['3rd_ranking_count'] + 1, feature_ranking['3rd_ranking_count'])
    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 3, feature_ranking['3rd_ranking_count'] + 1, feature_ranking['3rd_ranking_count'])
    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MI'] == 3, feature_ranking['3rd_ranking_count'] + 1, feature_ranking['3rd_ranking_count'])

    feature_ranking['Final_weight'] =  (feature_ranking['1st_ranking_count'] + feature_ranking['2nd_ranking_count'] + feature_ranking['3rd_ranking_count'])/3
    feature_ranking['Final_weight'] = np.where(feature_ranking['Final_weight'] < 1, 1, feature_ranking['Final_weight'])

    feature_ranking['Final_Filter'] = (feature_ranking['Variance_stand'] + feature_ranking['Multi_Collinearity_flipped'] + feature_ranking['Lap_Score_flipped'] + feature_ranking['Mutual_Information_Stand']) * feature_ranking['Final_weight']

    feature_ranking.sort_values(by = ['Final_Filter'], ascending = False, inplace = True)
    feature_ranking.reset_index(drop = True, inplace = True)

    return feature_ranking, feature_sim_data

def FS_Var_Col_Lap(features_df = None):
    feature_ranking_variance = pd.DataFrame()
    feature_ranking_multi_collinearity = pd.DataFrame()
    feature_ranking_lap_score = pd.DataFrame()
    feature_ranking_correlation = pd.DataFrame()
    feature_ranking_mutual_info = pd.DataFrame()
    feature_ranking = pd.DataFrame()

    variance_list = []
    multi_collinearity = []
    lap_score_list = []
    mutual_information = []

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
    feature_list = sim_data_feature_names

    #Check to see which features are discrete and which are continuous
    feature_discrete_check = pd.DataFrame()
    feature_discrete_check['Feature'] = sim_data_feature_names
    feature_discrete_check['Discrete'] = False

    for feature_dc in sim_data_feature_names:
        current_feature = list(feature_sim_data[feature_dc])
        if all(item in {0, 1} for item in current_feature):
            print('Value is discrete')
            feature_discrete_check['Discrete'] = np.where(feature_discrete_check['Feature'] == feature_dc, True, feature_discrete_check['Discrete'])
    

    #loop to calculate variance, multi collinearity and mutual info
    for i in range(num_features):
        current_feature_name = str(sim_data_feature_names[i])
        current_feature = list(feature_sim_data[current_feature_name])

        discrete = feature_discrete_check[feature_discrete_check['Feature'] == current_feature_name]['Discrete'].item()
        # For feature selection based on variance:
        curr_var = np.var(current_feature)

        # For feature selection based on multi_collinearity
        curr_multi_col = variance_inflation_factor(feature_sim_data.values, i)
        
        #For feature selection based on mutual information
        mi_list_for_curr_feature = []
        second_loop_features = copy.deepcopy(sim_data_feature_names)
        second_loop_features.remove(second_loop_features[i])
        for feature_mi in sim_data_feature_names:
            curr_feature_discrete = feature_discrete_check[feature_discrete_check['Feature'] == feature_mi]['Discrete'].item()
            # print(feature_mi, curr_feature_discrete, current_feature_name, discrete)
            if curr_feature_discrete:
                mi = mutual_info_classif(feature_sim_data[current_feature_name].to_frame(), feature_sim_data[feature_mi], discrete_features=[discrete])
            else:
                mi = mutual_info_regression(feature_sim_data[current_feature_name].to_frame(), feature_sim_data[feature_mi], discrete_features=[discrete])
            
            mi_list_for_curr_feature.append(mi)
            curr_feature_mi_summed = sum(mi_list_for_curr_feature)

        #Append all the calculated values
        variance_list.append(curr_var)
        multi_collinearity.append(curr_multi_col)
        mutual_information.append(curr_feature_mi_summed[0])

    print('mutula info print:',mutual_information) 

    #Calculate the lap score
    feature_np_array = feature_sim_data.to_numpy(copy=True)
    # print('Feature_array:', feature_np_array)
    lap_result = lap_score(feature_np_array)
    # print("Lap lacian score:",lap_result)    

    #Add feature names to each feature ranking dataframe
    feature_ranking_variance['Feature'] = feature_list
    feature_ranking_multi_collinearity['Feature'] = feature_list
    feature_ranking_lap_score['Feature'] = feature_list
    feature_ranking_mutual_info['Feature'] = feature_list

    #Create each feature ranking and sort them by their respecfeature_listtive selection method
    feature_ranking_variance['Variance'] = variance_list
    feature_ranking_variance['Variance'].replace([np.inf, -np.inf], 0, inplace=True)
    feature_ranking_variance['Variance_stand'] = list((feature_ranking_variance['Variance'] - np.min(feature_ranking_variance['Variance'])) / (np.max(feature_ranking_variance['Variance']) - np.min(feature_ranking_variance['Variance'])))
    feature_ranking_variance.sort_values(by = ['Variance_stand'], ascending = False, inplace = True)
    feature_ranking_variance.reset_index(drop = True, inplace = True)
    feature_ranking_variance['Feature_ranking_Va'] = 1
    feature_ranking_variance['Feature_ranking_Va'] = feature_ranking_variance['Feature_ranking_Va'].cumsum()
    print(feature_ranking_variance)

    feature_ranking_multi_collinearity['Multi_Collinearity'] = multi_collinearity
    feature_ranking_multi_collinearity['Multi_Collinearity'].replace([np.inf, -np.inf], 0, inplace=True)
    feature_ranking_multi_collinearity['Multi_Collinearity_stand'] = list((feature_ranking_multi_collinearity['Multi_Collinearity'] - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])) / (np.max(feature_ranking_multi_collinearity['Multi_Collinearity']) - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])))
    feature_ranking_multi_collinearity['Multi_Collinearity_flipped'] = 1 - feature_ranking_multi_collinearity['Multi_Collinearity_stand'] 
    feature_ranking_multi_collinearity.sort_values(by = ['Multi_Collinearity_flipped'], ascending = False, inplace = True)
    feature_ranking_multi_collinearity.reset_index(drop = True, inplace = True)
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = 1
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = feature_ranking_multi_collinearity['Feature_ranking_MC'].cumsum()
    print(feature_ranking_multi_collinearity)

    feature_ranking_lap_score['Lap_Score'] = lap_result
    feature_ranking_lap_score['Lap_Score_stand'] = list((feature_ranking_lap_score['Lap_Score'] - np.min(feature_ranking_lap_score['Lap_Score'])) / (np.max(feature_ranking_lap_score['Lap_Score']) - np.min(feature_ranking_lap_score['Lap_Score'])))
    feature_ranking_lap_score['Lap_Score_flipped'] = 1 - feature_ranking_lap_score['Lap_Score_stand'] 
    feature_ranking_lap_score.sort_values(by = ['Lap_Score_flipped'], ascending = False, inplace = True)
    feature_ranking_lap_score.reset_index(drop = True, inplace = True)
    feature_ranking_lap_score['Feature_ranking_LP'] = 1
    feature_ranking_lap_score['Feature_ranking_LP'] = feature_ranking_lap_score['Feature_ranking_LP'].cumsum()
    print(feature_ranking_lap_score)

    # Create the final feature ranking df
    feature_ranking = pd.merge(feature_ranking_multi_collinearity, feature_ranking_variance, how = 'left', on = ['Feature'])
    feature_ranking = pd.merge(feature_ranking, feature_ranking_lap_score, how = 'left', on = ['Feature'])

    #Create a ranking count to see overlap
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 1, 1, 0)
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 1, feature_ranking['1st_ranking_count'] + 1, feature_ranking['1st_ranking_count'])
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 1, feature_ranking['1st_ranking_count'] + 1, feature_ranking['1st_ranking_count'])

    feature_ranking['1st_ranking_count'] = feature_ranking['1st_ranking_count']*3

    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 2, 1, 0)
    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 2, feature_ranking['2nd_ranking_count'] + 1, feature_ranking['2nd_ranking_count'])
    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 2, feature_ranking['2nd_ranking_count'] + 1, feature_ranking['2nd_ranking_count'])
    feature_ranking['2nd_ranking_count'] = feature_ranking['2nd_ranking_count']*2

    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 3, 1, 0)
    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 3, feature_ranking['3rd_ranking_count'] + 1, feature_ranking['3rd_ranking_count'])
    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 3, feature_ranking['3rd_ranking_count'] + 1, feature_ranking['3rd_ranking_count'])

    feature_ranking['Final_weight'] =  (feature_ranking['1st_ranking_count'] + feature_ranking['2nd_ranking_count'] + feature_ranking['3rd_ranking_count'])/3
    feature_ranking['Final_weight'] = np.where(feature_ranking['Final_weight'] < 1, 1, feature_ranking['Final_weight'])

    feature_ranking['Final_Filter'] = (feature_ranking['Variance_stand'] + feature_ranking['Multi_Collinearity_flipped'] + feature_ranking['Lap_Score_flipped']) * feature_ranking['Final_weight']

    feature_ranking.sort_values(by = ['Final_Filter'], ascending = False, inplace = True)
    feature_ranking.reset_index(drop = True, inplace = True)

    return feature_ranking, feature_sim_data

def FS_Col_Lap(features_df = None):
    feature_ranking_variance = pd.DataFrame()
    feature_ranking_multi_collinearity = pd.DataFrame()
    feature_ranking_lap_score = pd.DataFrame()
    feature_ranking_correlation = pd.DataFrame()
    feature_ranking_mutual_info = pd.DataFrame()
    feature_ranking = pd.DataFrame()

    variance_list = []
    multi_collinearity = []
    lap_score_list = []
    mutual_information = []

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
    feature_list = sim_data_feature_names

    #Check to see which features are discrete and which are continuous
    feature_discrete_check = pd.DataFrame()
    feature_discrete_check['Feature'] = sim_data_feature_names
    feature_discrete_check['Discrete'] = False

    for feature_dc in sim_data_feature_names:
        current_feature = list(feature_sim_data[feature_dc])
        if all(item in {0, 1} for item in current_feature):
            print('Value is discrete')
            feature_discrete_check['Discrete'] = np.where(feature_discrete_check['Feature'] == feature_dc, True, feature_discrete_check['Discrete'])
    

    #loop to calculate variance, multi collinearity and mutual info
    for i in range(num_features):
        current_feature_name = str(sim_data_feature_names[i])
        current_feature = list(feature_sim_data[current_feature_name])

        discrete = feature_discrete_check[feature_discrete_check['Feature'] == current_feature_name]['Discrete'].item()
        # For feature selection based on multi_collinearity
        curr_multi_col = variance_inflation_factor(feature_sim_data.values, i)
        

        #Append all the calculated values
        multi_collinearity.append(curr_multi_col)

    print('mutula info print:',mutual_information) 

    #Calculate the lap score
    feature_np_array = feature_sim_data.to_numpy(copy=True)
    # print('Feature_array:', feature_np_array)
    lap_result = lap_score(feature_np_array)
    # print("Lap lacian score:",lap_result)    

    #Add feature names to each feature ranking dataframe
    feature_ranking_multi_collinearity['Feature'] = feature_list
    feature_ranking_lap_score['Feature'] = feature_list

    #Create each feature ranking and sort them by their respecfeature_listtive selection method

    feature_ranking_multi_collinearity['Multi_Collinearity'] = multi_collinearity
    feature_ranking_multi_collinearity['Multi_Collinearity'].replace([np.inf, -np.inf], 0, inplace=True)
    feature_ranking_multi_collinearity['Multi_Collinearity_stand'] = list((feature_ranking_multi_collinearity['Multi_Collinearity'] - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])) / (np.max(feature_ranking_multi_collinearity['Multi_Collinearity']) - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])))
    feature_ranking_multi_collinearity['Multi_Collinearity_flipped'] = 1 - feature_ranking_multi_collinearity['Multi_Collinearity_stand'] 
    feature_ranking_multi_collinearity.sort_values(by = ['Multi_Collinearity_flipped'], ascending = False, inplace = True)
    feature_ranking_multi_collinearity.reset_index(drop = True, inplace = True)
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = 1
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = feature_ranking_multi_collinearity['Feature_ranking_MC'].cumsum()
    print(feature_ranking_multi_collinearity)

    feature_ranking_lap_score['Lap_Score'] = lap_result
    feature_ranking_lap_score['Lap_Score_stand'] = list((feature_ranking_lap_score['Lap_Score'] - np.min(feature_ranking_lap_score['Lap_Score'])) / (np.max(feature_ranking_lap_score['Lap_Score']) - np.min(feature_ranking_lap_score['Lap_Score'])))
    feature_ranking_lap_score['Lap_Score_flipped'] = 1 - feature_ranking_lap_score['Lap_Score_stand'] 
    feature_ranking_lap_score.sort_values(by = ['Lap_Score_flipped'], ascending = False, inplace = True)
    feature_ranking_lap_score.reset_index(drop = True, inplace = True)
    feature_ranking_lap_score['Feature_ranking_LP'] = 1
    feature_ranking_lap_score['Feature_ranking_LP'] = feature_ranking_lap_score['Feature_ranking_LP'].cumsum()
    print(feature_ranking_lap_score)

    # Create the final feature ranking df
    feature_ranking = pd.merge(feature_ranking_multi_collinearity, feature_ranking_lap_score, how = 'left', on = ['Feature'])

    #Create a ranking count to see overlap
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 1, 1, 0)
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 1, feature_ranking['1st_ranking_count'] + 1, feature_ranking['1st_ranking_count'])

    feature_ranking['1st_ranking_count'] = feature_ranking['1st_ranking_count']*3

    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 2, 1, 0)
    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 2, feature_ranking['2nd_ranking_count'] + 1, feature_ranking['2nd_ranking_count'])
    feature_ranking['2nd_ranking_count'] = feature_ranking['2nd_ranking_count']*2

    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 3, 1, 0)
    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_LP'] == 3, feature_ranking['3rd_ranking_count'] + 1, feature_ranking['3rd_ranking_count'])

    feature_ranking['Final_weight'] =  (feature_ranking['1st_ranking_count'] + feature_ranking['2nd_ranking_count'] + feature_ranking['3rd_ranking_count'])/3
    feature_ranking['Final_weight'] = np.where(feature_ranking['Final_weight'] < 1, 1, feature_ranking['Final_weight'])

    feature_ranking['Final_Filter'] = (feature_ranking['Multi_Collinearity_flipped'] + feature_ranking['Lap_Score_flipped']) * feature_ranking['Final_weight']

    feature_ranking.sort_values(by = ['Final_Filter'], ascending = False, inplace = True)
    feature_ranking.reset_index(drop = True, inplace = True)

    return feature_ranking, feature_sim_data

def FS_Var_Col(features_df = None):
    feature_ranking_variance = pd.DataFrame()
    feature_ranking_multi_collinearity = pd.DataFrame()
    feature_ranking_lap_score = pd.DataFrame()
    feature_ranking_correlation = pd.DataFrame()
    feature_ranking_mutual_info = pd.DataFrame()
    feature_ranking = pd.DataFrame()

    variance_list = []
    multi_collinearity = []
    lap_score_list = []
    mutual_information = []

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
    feature_list = sim_data_feature_names

    #Check to see which features are discrete and which are continuous
    feature_discrete_check = pd.DataFrame()
    feature_discrete_check['Feature'] = sim_data_feature_names
    feature_discrete_check['Discrete'] = False

    for feature_dc in sim_data_feature_names:
        current_feature = list(feature_sim_data[feature_dc])
        if all(item in {0, 1} for item in current_feature):
            print('Value is discrete')
            feature_discrete_check['Discrete'] = np.where(feature_discrete_check['Feature'] == feature_dc, True, feature_discrete_check['Discrete'])
    

    #loop to calculate variance, multi collinearity and mutual info
    for i in range(num_features):
        current_feature_name = str(sim_data_feature_names[i])
        current_feature = list(feature_sim_data[current_feature_name])

        discrete = feature_discrete_check[feature_discrete_check['Feature'] == current_feature_name]['Discrete'].item()
        # For feature selection based on variance:
        curr_var = np.var(current_feature)

        # For feature selection based on multi_collinearity
        curr_multi_col = variance_inflation_factor(feature_sim_data.values, i)
        
        #Append all the calculated values
        variance_list.append(curr_var)
        multi_collinearity.append(curr_multi_col)

    print('mutula info print:',mutual_information) 

    #Add feature names to each feature ranking dataframe
    feature_ranking_variance['Feature'] = feature_list
    feature_ranking_multi_collinearity['Feature'] = feature_list

    #Create each feature ranking and sort them by their respecfeature_listtive selection method
    feature_ranking_variance['Variance'] = variance_list
    feature_ranking_variance['Variance'].replace([np.inf, -np.inf], 0, inplace=True)
    feature_ranking_variance['Variance_stand'] = list((feature_ranking_variance['Variance'] - np.min(feature_ranking_variance['Variance'])) / (np.max(feature_ranking_variance['Variance']) - np.min(feature_ranking_variance['Variance'])))
    feature_ranking_variance.sort_values(by = ['Variance_stand'], ascending = False, inplace = True)
    feature_ranking_variance.reset_index(drop = True, inplace = True)
    feature_ranking_variance['Feature_ranking_Va'] = 1
    feature_ranking_variance['Feature_ranking_Va'] = feature_ranking_variance['Feature_ranking_Va'].cumsum()
    print(feature_ranking_variance)

    feature_ranking_multi_collinearity['Multi_Collinearity'] = multi_collinearity
    feature_ranking_multi_collinearity['Multi_Collinearity'].replace([np.inf, -np.inf], 0, inplace=True)
    feature_ranking_multi_collinearity['Multi_Collinearity_stand'] = list((feature_ranking_multi_collinearity['Multi_Collinearity'] - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])) / (np.max(feature_ranking_multi_collinearity['Multi_Collinearity']) - np.min(feature_ranking_multi_collinearity['Multi_Collinearity'])))
    feature_ranking_multi_collinearity['Multi_Collinearity_flipped'] = 1 - feature_ranking_multi_collinearity['Multi_Collinearity_stand'] 
    feature_ranking_multi_collinearity.sort_values(by = ['Multi_Collinearity_flipped'], ascending = False, inplace = True)
    feature_ranking_multi_collinearity.reset_index(drop = True, inplace = True)
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = 1
    feature_ranking_multi_collinearity['Feature_ranking_MC'] = feature_ranking_multi_collinearity['Feature_ranking_MC'].cumsum()
    print(feature_ranking_multi_collinearity)

    # Create the final feature ranking df
    feature_ranking = pd.merge(feature_ranking_multi_collinearity, feature_ranking_variance, how = 'left', on = ['Feature'])

    #Create a ranking count to see overlap
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 1, 1, 0)
    feature_ranking['1st_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 1, feature_ranking['1st_ranking_count'] + 1, feature_ranking['1st_ranking_count'])

    feature_ranking['1st_ranking_count'] = feature_ranking['1st_ranking_count']*3

    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 2, 1, 0)
    feature_ranking['2nd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 2, feature_ranking['2nd_ranking_count'] + 1, feature_ranking['2nd_ranking_count'])
    feature_ranking['2nd_ranking_count'] = feature_ranking['2nd_ranking_count']*2

    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_Va'] == 3, 1, 0)
    feature_ranking['3rd_ranking_count'] = np.where(feature_ranking['Feature_ranking_MC'] == 3, feature_ranking['3rd_ranking_count'] + 1, feature_ranking['3rd_ranking_count'])

    feature_ranking['Final_weight'] =  (feature_ranking['1st_ranking_count'] + feature_ranking['2nd_ranking_count'] + feature_ranking['3rd_ranking_count'])/3
    feature_ranking['Final_weight'] = np.where(feature_ranking['Final_weight'] < 1, 1, feature_ranking['Final_weight'])

    feature_ranking['Final_Filter'] = (feature_ranking['Variance_stand'] + feature_ranking['Multi_Collinearity_flipped']) * feature_ranking['Final_weight']

    feature_ranking.sort_values(by = ['Final_Filter'], ascending = False, inplace = True)
    feature_ranking.reset_index(drop = True, inplace = True)

    return feature_ranking, feature_sim_data