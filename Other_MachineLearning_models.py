#Other machine learning models
    #LASSO
    #Ridge
    #Random Forest

import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import scipy
import time
from sklearn.decomposition import FactorAnalysis
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

#data preprocessing
descriptors_data = pd.read_csv('HC50raw.csv')
descriptors_data_processed = descriptors_data.dropna(axis=1)
HC50 = descriptors_data_processed.iloc[:, 1]
HC50 = np.array(HC50)
descriptors = descriptors_data_processed.iloc[:, 2:]
descriptors = np.array(descriptors)
n_sample = len(descriptors)
total_id = np.arange(n_sample)

def Kfold(length,fold):
    size = np.arange(length).tolist()
    train_index = []
    val_index = []
    rest = length % fold
    fold_size = int(length/fold)
    temp_fold_size = fold_size
    for i in range(fold):
        temp_train = []
        temp_val = []
        if rest>0:
            temp_fold_size = fold_size+1
            rest = rest -1
            temp_val = size[i*temp_fold_size:+i*temp_fold_size+temp_fold_size]
            temp_train = size[0:i*temp_fold_size] + size[i*temp_fold_size+temp_fold_size:]
        else:
            temp_val = size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size
                            :(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size]
            temp_train = size[0:(length % fold)*temp_fold_size+(i-(length % fold))*fold_size] + size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size:]
        train_index.append(temp_train)
        val_index.append(temp_val)
    return (train_index,val_index)

###################### LASSO ##########################

start=time.time()
r2_lasso_all = []
MAE_lasso_all = []
RMSE_lasso_all = []
MAPE_lasso_all = []
for t in range(5):
    train_split_index,test_split_index = Kfold(len(HC50),5)
    np.random.shuffle(total_id)
    splits = 5
    prediction_lasso = []
    prediction_true_lasso = []
    test_score_all_lasso = []

    for k in range(splits):

        print('batch is ',k)
        train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
        test_index = test_split_index[k]


        train_id = [total_id[i] for i in train_index]
        valid_id = [total_id[i] for i in valid_index]
        test_id = [total_id[i] for i in test_index]

        train_feature = np.array([descriptors[i] for i in train_id])
        train_label = np.array([HC50[i] for i in train_id])
        valid_feature = np.array([descriptors[i] for i in valid_id])
        valid_label = np.array([HC50[i] for i in valid_id])
        test_feature = np.array([descriptors[i] for i in test_id])
        test_label = np.array([HC50[i] for i in test_id])

        t_v = np.concatenate((train_feature,valid_feature),0)
        mean = np.mean(t_v,0)
        std = np.std(t_v,0)

        train_feature = (train_feature - mean)/std
        valid_feature = (valid_feature - mean)/std
        test_feature = (test_feature - mean)/std

        train_feature[np.isnan(train_feature)] = 0
        valid_feature[np.isnan(valid_feature)] = 0
        test_feature[np.isnan(test_feature)] = 0

        train_feature[np.isinf(train_feature)] = 0
        valid_feature[np.isinf(valid_feature)] = 0
        test_feature[np.isinf(test_feature)] = 0

        alpha_pool = [0.1,0.01,0.005,0.05,1,10,100]

        best_valid_score = 0
        for a in alpha_pool:
            model = linear_model.Lasso(alpha=a)
            model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
            valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
            #print(valid_score)
            if valid_score>best_valid_score:
                best_valid_score = valid_score
                test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                pred = model.predict(test_feature)
                best_a = a
                best_MAE = np.mean(np.abs(pred.reshape(-1)-np.array(test_label).reshape(-1)))

        print(test_score)
        prediction_lasso.append(pred)
        prediction_true_lasso.append(test_label)
        test_score_all_lasso.append(test_score)
        print('best a is',best_a)
        print('best mae',best_MAE)
        #print('feature importance',model.feature_importances_)


    prediction_lasso_all = []
    for l in prediction_lasso:
        for v in l:
            prediction_lasso_all.append(v)

    prediction_true_lasso_all = []
    for l in prediction_true_lasso:
        for v in l:
            prediction_true_lasso_all.append(v)


    MAE = np.mean(np.abs(np.array(prediction_true_lasso_all)-np.array(prediction_lasso_all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_lasso_all), np.array(prediction_lasso_all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_lasso_all), np.array(prediction_lasso_all).reshape(-1))

    r2_lasso_all.append(np.mean(test_score_all_lasso))
    MAE_lasso_all.append(MAE)
    RMSE_lasso_all.append(RMSE)
    MAPE_lasso_all.append(MAPE)
end = time.time()

#end-start

d1={'rsquared':str(np.mean(r2_lasso_all).round(3))+u"\u00B1"+str(np.std(r2_lasso_all).round(3)),
    'MAE':str(np.mean(MAE_lasso_all).round(3))+u"\u00B1"+str(np.std(MAE_lasso_all).round(3)),
    'RMSE':str(np.mean(RMSE_lasso_all).round(3))+u"\u00B1"+str(np.std(RMSE_lasso_all).round(3))
   }
OUTPUTLASSO= pd.DataFrame([d1])
print('LASSO results is',OUTPUTLASSO)

####################### Ridge ########################

#featurez
start = time.time()
r2_ridge_all = []
MAE_ridge_all = []
RMSE_ridge_all= []
MAPE_ridge_all = []

for t in range(5): 
    print('repeat run',t)
    train_split_index,test_split_index = Kfold(len(HC50),5)
    np.random.shuffle(total_id)
    splits = 5
    prediction_ridge = []
    prediction_true_ridge = []
    test_score_all_ridge = []

    for k in range(splits):

        print('batch is ',k)
        train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
        test_index = test_split_index[k]


        train_id = [total_id[i] for i in train_index]
        valid_id = [total_id[i] for i in valid_index]
        test_id = [total_id[i] for i in test_index]

        train_feature = np.array([descriptors[i] for i in train_id])
        train_label = np.array([HC50[i] for i in train_id])
        valid_feature = np.array([descriptors[i] for i in valid_id])
        valid_label = np.array([HC50[i] for i in valid_id])
        test_feature = np.array([descriptors[i] for i in test_id])
        test_label = np.array([HC50[i] for i in test_id])

        t_v = np.concatenate((train_feature,valid_feature),0)
        mean = np.mean(t_v,0)
        std = np.std(t_v,0)

        train_feature = (train_feature - mean)/std
        valid_feature = (valid_feature - mean)/std
        test_feature = (test_feature - mean)/std

        train_feature[np.isnan(train_feature)] = 0
        valid_feature[np.isnan(valid_feature)] = 0
        test_feature[np.isnan(test_feature)] = 0

        train_feature[np.isinf(train_feature)] = 0
        valid_feature[np.isinf(valid_feature)] = 0
        test_feature[np.isinf(test_feature)] = 0

        alpha_pool = [0.1,0.001,0.01,1,10,100,200,500,750]

        best_valid_score = 0
        for a in alpha_pool:
            model = Ridge(alpha=a)
            model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
            valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
            #print(valid_score)
            if valid_score>best_valid_score:
                best_valid_score = valid_score
                test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                pred = model.predict(test_feature)
                best_a = a
                best_MAE = np.mean(np.abs(pred.reshape(-1)-np.array(test_label).reshape(-1)))

        print(test_score)
        prediction_ridge.append(pred)
        prediction_true_ridge.append(test_label)
        test_score_all_ridge.append(test_score)
        print('best a is',best_a)
        print('best mae',best_MAE)
        #print('feature importance',model.feature_importances_)

    prediction_ridge_all = []
    for l in prediction_ridge:
        for v in l:
            prediction_ridge_all.append(v)

    prediction_true_ridge_all = []
    for l in prediction_true_ridge:
        for v in l:
            prediction_true_ridge_all.append(v)


    MAE = np.mean(np.abs(np.array(prediction_true_ridge_all)-np.array(prediction_ridge_all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_ridge_all), np.array(prediction_ridge_all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_ridge_all), np.array(prediction_ridge_all).reshape(-1))

    r2_ridge_all.append(np.mean(test_score_all_ridge))
    MAE_ridge_all.append(MAE)
    RMSE_ridge_all.append(RMSE)
    MAPE_ridge_all.append(MAPE)
    
end = time.time()

#end-start

d1={'rsquared':str(np.mean(r2_ridge_all).round(3))+u"\u00B1"+str(np.std(r2_ridge_all).round(3)),
    'MAE':str(np.mean(MAE_ridge_all).round(3))+u"\u00B1"+str(np.std(MAE_ridge_all).round(3)),
    'RMSE':str(np.mean(RMSE_ridge_all).round(3))+u"\u00B1"+str(np.std(RMSE_ridge_all).round(3))
   }
OUTPUTridge= pd.DataFrame([d1])
print('ridge result is',OUTPUTridge)


######################### Random Forest#######################
start = time.time()
r2_rf_all = []
MAE_rf_all = []
RMSE_rf_all = []
MAPE_rf_all = []

for t in range(5):
    train_split_index,test_split_index = Kfold(len(HC50),5)
    np.random.shuffle(total_id)
    splits = 5
    prediction_rf = []
    prediction_true_rf = []
    test_score_all_rf = []

    for k in range(splits):

        print('batch is ',k)
        train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
        test_index = test_split_index[k]

        train_id = [total_id[i] for i in train_index]
        valid_id = [total_id[i] for i in valid_index]
        test_id = [total_id[i] for i in test_index]

        train_feature = [descriptors[i] for i in train_id]
        train_label = [HC50[i] for i in train_id]

        valid_feature = [descriptors[i] for i in valid_id]
        valid_label = [HC50[i] for i in valid_id]

        test_feature = [descriptors[i] for i in test_id]
        test_label = [HC50[i] for i in test_id]

        n_estimator = [100,200,300,500,1000]
        #400,600
        #max_depths = [10,11]
        max_features = ['auto', 'sqrt', 'log2']
        best_valid_score = 0
        for ne in n_estimator:
            for m_d in max_features:
                model = RandomForestRegressor(n_estimators=ne,max_features=m_d)
                model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
                valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
                #print(valid_score)
                if valid_score>best_valid_score:
                    best_valid_score = valid_score
                    test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                    pred = model.predict(test_feature)
                    best_n = ne
                    best_d = m_d
                    best_MAE = np.mean(np.abs(pred.reshape(-1)-np.array(test_label).reshape(-1)))

        print(test_score)
        prediction_rf.append(pred)
        prediction_true_rf.append(test_label)
        test_score_all_rf.append(test_score)
        print('best n_estimator is',best_n)
        print('best feature is',best_d)
        print('best mae',best_MAE)
        #print('feature importance',model.feature_importances_)
    print('this repeated run',np.mean(test_score_all_rf))
    prediction_rf_all = []
    for l in prediction_rf:
        for v in l:
            prediction_rf_all.append(v)

    prediction_true_rf_all = []
    for l in prediction_true_rf:
        for v in l:
            prediction_true_rf_all.append(v)

    MAE = np.mean(np.abs(np.array(prediction_true_rf_all)-np.array(prediction_rf_all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_rf_all), np.array(prediction_rf_all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_rf_all), np.array(prediction_rf_all).reshape(-1))

    r2_rf_all.append(np.mean(test_score_all_rf))
    MAE_rf_all.append(MAE)
    RMSE_rf_all.append(RMSE)
    MAPE_rf_all.append(MAPE)
end=time.time()

print('random forest time is',(end-start)/60)

drf={'rsquared':str(np.mean(r2_rf_all).round(3))+u"\u00B1"+str(np.std(r2_rf_all).round(3)),
    'MAE':str(np.mean(MAE_rf_all).round(3))+u"\u00B1"+str(np.std(MAE_rf_all).round(3)),
    'RMSE':str(np.mean(RMSE_rf_all).round(3))+u"\u00B1"+str(np.std(RMSE_rf_all).round(3))
   }
OUTPUTrf= pd.DataFrame([drf])
print('random forest result is', OUTPUTrf)


# In[ ]:




