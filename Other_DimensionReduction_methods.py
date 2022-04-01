#Other dimension reduction methods, includes:
    #PCA
    #Kernel PCA
    #UMAP
    #Factor analysis
    #MDS

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import scipy
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS
import umap
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import FactorAnalysis

#data preprocessing
descriptors_data = pd.read_csv('HC50raw.csv')
descriptors_data_processed = descriptors_data.dropna(axis=1)
HC50 = descriptors_data_processed.iloc[:, 1]
HC50 = np.array(HC50)
descriptors = descriptors_data_processed.iloc[:, 2:]
descriptors = np.array(descriptors)
n_sample = len(descriptors)

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

############### PCA128###################
start = time.time()
r2_pca_all=[]
MAE_pca_all = []
RMSE_pca_all = []
MAPE_pca_all = []

for t in range(5):
    train_split_index,test_split_index = Kfold(n_sample,5)
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    splits = 5
    prediction_pca = []
    prediction_true_pca = []
    test_score_pca_all = []
    for k in range(splits):

        print('batch is ',k)
        pca128_operator = PCA(n_components=128)

        train_id = [total_id[i] for i in train_split_index[k]]
        test_id = [total_id[i] for i in test_split_index[k]]

        train_feature = np.array([descriptors[i] for i in train_id])
        train_label = np.array([HC50[i] for i in train_id])
        
        test_feature = np.array([descriptors[i] for i in test_id])
        test_label = np.array([HC50[i] for i in test_id])

        mean = np.mean(train_feature,0)
        std = np.std(train_feature,0)

        train_feature = (np.array(train_feature) - mean)/std
        test_feature = (np.array(test_feature) - mean)/std

        train_feature[np.isnan(train_feature)] = 0
        test_feature[np.isnan(test_feature)] = 0

        train_feature[np.isinf(train_feature)] = 0
        test_feature[np.isinf(test_feature)] = 0
        
        pca128_operator.fit(train_feature)
        train_feature_ = pca128_operator.transform(train_feature)
        test_feature_ = pca128_operator.transform(test_feature)

        model = LinearRegression()
        model.fit(np.array(train_feature_),np.array(train_label).reshape(-1))
        test_score = model.score(test_feature_,np.array(test_label).reshape(-1,1))
        pred = model.predict(test_feature_)

        print(test_score)
        prediction_pca.append(pred)
        prediction_true_pca.append(test_label)
        test_score_pca_all.append(test_score)


    prediction_pca_all = []
    for l in prediction_pca:
        for v in l:
            prediction_pca_all.append(v)

    prediction_true_pca_all = []
    for l in prediction_true_pca:
        for v in l:
            prediction_true_pca_all.append(v)

    MAE = np.mean(np.abs(np.array(prediction_true_pca_all)-np.array(prediction_pca_all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_pca_all), np.array(prediction_pca_all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_pca_all), np.array(prediction_pca_all).reshape(-1))

    r2_pca_all.append(np.mean(test_score_pca_all))
    MAE_pca_all.append(MAE)
    RMSE_pca_all.append(RMSE)
    MAPE_pca_all.append(MAPE)
end = time.time()

print('time is',(end-start))

dPCA128={'rsquared':str(np.mean(r2_pca_all).round(3))+u"\u00B1"+str(np.std(r2_pca_all).round(3)),
    'MAE':str(np.mean(MAE_pca_all).round(3))+u"\u00B1"+str(np.std(MAE_pca_all).round(3)),
    'RMSE':str(np.mean(RMSE_pca_all).round(3))+u"\u00B1"+str(np.std(RMSE_pca_all).round(3))
   }
OUTPUT= pd.DataFrame([dPCA128])
print('PCAresults is',OUTPUT)


############### kernel PCA128###################
start = time.time()
MAE_kPCA_all = []
RMSE_kPCA_all = []
MAPE_kPCA_all = []
r2_kPCA_all = []

for t in range(5):
    train_split_index,test_split_index = Kfold(n_sample,5)
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    splits = 5
    prediction_kpca = []
    prediction_true_kpca= []
    test_score_all_kpca = []
    for k in range(splits):

        print('batch is ',k)
        kpca128_operator = KernelPCA(n_components=128, kernel='rbf')

        train_id = [total_id[i] for i in train_split_index[k]]
        test_id = [total_id[i] for i in test_split_index[k]]

        train_feature = np.array([descriptors[i] for i in train_id])
        train_label = np.array([HC50[i] for i in train_id])
        
        test_feature = np.array([descriptors[i] for i in test_id])
        test_label = np.array([HC50[i] for i in test_id])

        mean = np.mean(train_feature,0)
        std = np.std(train_feature,0)

        train_feature = (train_feature - mean)/std
        test_feature = (test_feature - mean)/std

        train_feature[np.isnan(train_feature)] = 0
        test_feature[np.isnan(test_feature)] = 0

        train_feature[np.isinf(train_feature)] = 0
        test_feature[np.isinf(test_feature)] = 0
        
        kpca128_operator.fit(train_feature)
        train_feature_ = kpca128_operator.transform(train_feature)
        test_feature_ = kpca128_operator.transform(test_feature)

        model = LinearRegression()
        model.fit(np.array(train_feature_),np.array(train_label).reshape(-1))
        test_score = model.score(test_feature_,np.array(test_label).reshape(-1,1))
        pred = model.predict(test_feature_)

        print(test_score)
        prediction_kpca.append(pred)
        prediction_true_kpca.append(test_label)
        test_score_all_kpca.append(test_score)


    prediction_kPCA_all = []
    for l in prediction_kpca:
        for v in l:
            prediction_kPCA_all.append(v)

    prediction_true_kPCA_all = []
    for l in prediction_true_kpca:
        for v in l:
            prediction_true_kPCA_all.append(v)

    MAE = np.mean(np.abs(np.array(prediction_true_kPCA_all)-np.array(prediction_kPCA_all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_kPCA_all), np.array(prediction_kPCA_all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_kPCA_all), np.array(prediction_kPCA_all).reshape(-1))

    r2_kPCA_all.append(np.mean(test_score_all_kpca))
    MAE_kPCA_all.append(MAE)
    RMSE_kPCA_all.append(RMSE)
    MAPE_kPCA_all.append(MAPE)

d1={'rsquared':str(np.mean(r2_kPCA_all).round(3))+u"\u00B1"+str(np.std(r2_kPCA_all).round(3)),
    'MAE':str(np.mean(MAE_kPCA_all).round(3))+u"\u00B1"+str(np.std(MAE_kPCA_all).round(3)),
    'RMSE':str(np.mean(RMSE_kPCA_all).round(3))+u"\u00B1"+str(np.std(RMSE_kPCA_all).round(3))
   }
OUTPUTkernelPCA= pd.DataFrame([d1])
print('OUTPUTkernelPCA result is', OUTPUTkernelPCA)


######################## UMAP128######################
#umap sometimes give negative or ~0 r2 results, which could due to bad training/test splits
#for comparison, we dropped those negative results
#this will overestimate (intentionally for comparison) umap performance
start=time.time()
r2_umap_all=[]
MAE_umap_all = []
RMSE_umap_all = []
MAPE_umap_all = []

for t in range(5):
    train_split_index,test_split_index = Kfold(n_sample,5)
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    splits = 5
    prediction_umap = []
    prediction_true_umap = []
    test_score_umap_all = []
    for k in range(splits):

        print('batch is ',k)
        umap128_operator = umap.UMAP(n_components=128)
        
        train_id = [total_id[i] for i in train_split_index[k]]
        test_id = [total_id[i] for i in test_split_index[k]]

        train_feature = [descriptors[i] for i in train_id]
        train_label = [HC50[i] for i in train_id]

        test_feature = [descriptors[i] for i in test_id]
        test_label = [HC50[i] for i in test_id]
        
        mean = np.mean(train_feature,0)
        std = np.std(train_feature,0)

        train_feature = (np.array(train_feature) - mean)/std
        test_feature = (np.array(test_feature) - mean)/std

        train_feature[np.isnan(train_feature)] = 0
        test_feature[np.isnan(test_feature)] = 0

        train_feature[np.isinf(train_feature)] = 0
        test_feature[np.isinf(test_feature)] = 0
        
        umap128_operator.fit(train_feature)
        
        train_feature_ = umap128_operator.transform(train_feature)
        test_feature_ = umap128_operator.transform(test_feature)
        model = LinearRegression()
        model.fit(np.array(train_feature_),np.array(train_label).reshape(-1))
        test_score = model.score(test_feature_,np.array(test_label).reshape(-1,1))
        pred = model.predict(test_feature_)

        print(test_score)
        prediction_umap.append(pred)
        prediction_true_umap.append(test_label)
        test_score_umap_all.append(test_score)


    prediction_umap_all = []
    for l in prediction_umap:
        for v in l:
            prediction_umap_all.append(v)

    prediction_true_umap_all = []
    for l in prediction_true_umap:
        for v in l:
            prediction_true_umap_all.append(v)

    MAE = np.mean(np.abs(np.array(prediction_true_umap_all)-np.array(prediction_umap_all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_umap_all), np.array(prediction_umap_all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_umap_all), np.array(prediction_umap_all).reshape(-1))

    r2_umap_all.append(np.mean(test_score_umap_all))
    MAE_umap_all.append(MAE)
    RMSE_umap_all.append(RMSE)
    MAPE_umap_all.append(MAPE)
end=time.time()

print('UMAP time',end-start)

dUMAP={'rsquared':str(np.mean(r2_umap_all).round(3))+u"\u00B1"+str(np.std(r2_umap_all).round(3)),
    'MAE':str(np.mean(MAE_umap_all).round(3))+u"\u00B1"+str(np.std(MAE_umap_all).round(3)),
    'RMSE':str(np.mean(RMSE_umap_all).round(3))+u"\u00B1"+str(np.std(RMSE_umap_all).round(3))
   }
OUTPUTUMAP= pd.DataFrame([dUMAP])
print('UMAP128results is',OUTPUTUMAP)


##################################Factor Analysis##############################
MAE_FA_all = []
RMSE_FA_all = []
MAPE_FA_all = []
r2_FA_all = []
for t in range(5):
    print('repeat run',t)
    np.random.shuffle(total_id)
    train_split_index,test_split_index = Kfold(n_sample,5)

    splits = 5
    prediction_fa = []
    prediction_true_fa= []
    test_score_all_fa = []
    for k in range(splits):

        print('split is ',k)
        FA_operator = FactorAnalysis(n_components=128)

        train_id = [total_id[i] for i in train_split_index[k]]
        test_id = [total_id[i] for i in test_split_index[k]]

        train_feature = [descriptors[i] for i in train_id]
        train_label = [HC50[i] for i in train_id]

        test_feature = [descriptors[i] for i in test_id]
        test_label = [HC50[i] for i in test_id]
        
        mean = np.mean(train_feature,0)
        std = np.std(train_feature,0)
        
        train_feature = (np.array(train_feature) - mean)/std
        test_feature = (np.array(test_feature) - mean)/std

        train_feature[np.isnan(train_feature)] = 0
        test_feature[np.isnan(test_feature)] = 0

        train_feature[np.isinf(train_feature)] = 0
        test_feature[np.isinf(test_feature)] = 0
        
        FA_operator.fit(train_feature)

        train_feature_ = FA_operator.transform(train_feature)
        test_feature_ = FA_operator.transform(test_feature)
        
        model = LinearRegression()
        model.fit(np.array(train_feature_),np.array(train_label).reshape(-1,1))
        test_score = model.score(np.array(test_feature_),np.array(test_label).reshape(-1,1))
        pred = model.predict(test_feature_)

        print(test_score)
        prediction_fa.append(pred)
        prediction_true_fa.append(test_label)
        test_score_all_fa.append(test_score)

    prediction_fa_all = []
    for l in prediction_fa:
        for v in l:
            prediction_fa_all.append(v)

    prediction_true_fa_all = []
    for l in prediction_true_fa:
        for v in l:
            prediction_true_fa_all.append(v)
            
    MAE = np.mean(np.abs(np.array(prediction_true_fa_all)-np.array(prediction_fa_all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_fa_all), np.array(prediction_fa_all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_fa_all), np.array(prediction_fa_all).reshape(-1))
    
    r2_FA_all.append(np.mean(test_score_all_fa))
    MAE_FA_all.append(MAE)
    RMSE_FA_all.append(RMSE)
    MAPE_FA_all.append(MAPE)

d1={'rsquared':str(np.mean(r2_FA_all).round(3))+u"\u00B1"+str(np.std(r2_FA_all).round(3)),
    'MAE':str(np.mean(MAE_FA_all).round(3))+u"\u00B1"+str(np.std(MAE_FA_all).round(3)),
    'RMSE':str(np.mean(RMSE_FA_all).round(3))+u"\u00B1"+str(np.std(RMSE_FA_all).round(3))
   }
OUTPUTFA= pd.DataFrame([d1])
print('factor analysis results is',OUTPUTFA)


#################### MDS128##########################

#mds needs calculation of pair-wise distance and doesn't have a single "transform" function for test data
#we hence directly applied it on the whole dataset, this will expose training data to test data, i.e., overestimate the mds performance
feature_z = scipy.stats.mstats.zscore(descriptors,0)
embedding_MDS = MDS(n_components=128).fit_transform(feature_z)

MAE_mds_all = []
RMSE_mds_all = []
MAPE_mds_all = []
r2_mds_all = []

for t in range(5):
    train_split_index,test_split_index = Kfold(n_sample,5)
    np.random.shuffle(total_id)
    splits = 5
    prediction_mds = []
    prediction_true_mds= []
    test_score_all_mds = []
    for k in range(splits):

        print('batch is ',k)
        train_id = [total_id[i] for i in train_split_index[k]]
        test_id = [total_id[i] for i in test_split_index[k]]

        train_feature = np.array([embedding_MDS[i] for i in train_id])
        train_label = np.array([HC50[i] for i in train_id])

        test_feature = np.array([embedding_MDS[i] for i in test_id])
        test_label = np.array([HC50[i] for i in test_id])

        model = LinearRegression()
        model.fit(np.array(train_feature),np.array(train_label).reshape(-1,1))
        test_score = model.score(np.array(test_feature),np.array(test_label).reshape(-1,1))
        pred = model.predict(test_feature)

        print(test_score)
        prediction_mds.append(pred)
        prediction_true_mds.append(test_label)
        test_score_all_mds.append(test_score)

        prediction_mds_all = []
        for l in prediction_mds:
            for v in l:
                prediction_mds_all.append(v)

        prediction_true_mds_all = []
        for l in prediction_true_mds:
            for v in l:
                prediction_true_mds_all.append(v)

    MAE = np.mean(np.abs(np.array(prediction_true_mds_all)-np.array(prediction_mds_all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_mds_all), np.array(prediction_mds_all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_mds_all), np.array(prediction_mds_all).reshape(-1))

    r2_mds_all.append(np.mean(test_score_all_mds))
    MAE_mds_all.append(MAE)
    RMSE_mds_all.append(RMSE)
    MAPE_mds_all.append(MAPE)

d1={'rsquared':str(np.mean(r2_mds_all).round(3))+u"\u00B1"+str(np.std(r2_mds_all).round(3)),
    'MAE':str(np.mean(MAE_mds_all).round(3))+u"\u00B1"+str(np.std(MAE_mds_all).round(3)),
    'RMSE':str(np.mean(RMSE_mds_all).round(3))+u"\u00B1"+str(np.std(RMSE_mds_all).round(3))
   }
OUTPUTMDS= pd.DataFrame([d1])
print('MDS results is', OUTPUTMDS)

