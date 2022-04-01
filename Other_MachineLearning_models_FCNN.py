#Other machine learning models
    #FCNN

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import umap
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


#data preprocessing
descriptors_data = pd.read_csv('HC50raw.csv')
descriptors_data_processed = descriptors_data.dropna(axis=1)
HC50 = descriptors_data_processed.iloc[:, 1]
HC50 = np.array(HC50)
descriptors = descriptors_data_processed.iloc[:, 2:]
descriptors = np.array(descriptors)
n_sample = len(descriptors)
total_id = np.arange(n_sample)


class FCNN(nn.Module):
    def __init__(self,in_fea,h1,h2,latent_size,d_output,drop_rate):
        super(FCNN,self).__init__()

        self.l1 = nn.Linear(in_fea,h1)
        self.l3 = nn.Linear(h1,d_output)
        
    def forward(self,x):
        x1 = F.relu(self.l1(x))
        embedding = self.l3(x1)
        
        return embedding

class HC50Dataset(Dataset):

    def __init__(self,ids,preprocessed_data,labels):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = preprocessed_data[ids]
        self.label = labels[ids]
        self.id = ids
    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        label = self.label[idx]
        return sample,label

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

def train(model, iterator, optimizer,criterion2):
    
    model.train()
    loss = 0
    # Training
    train_MAE = 0
    train_r2 = 0
    embedding_all = []
    label_all = []
    for train_batch, train_label in iterator:
        # Model computations
        optimizer.zero_grad()

        prediction = model(train_batch)
        label_all.append(train_label)
        
        
        train_prediction_loss = criterion2(prediction.view(-1), train_label)
        MAE_temp = np.mean(np.abs(prediction.view(-1).detach().numpy()-train_label.detach().numpy()))
        r2_temp = r2_score(train_label.detach().numpy(),prediction.view(-1).detach().numpy())

        train_loss = train_prediction_loss
        train_loss.backward()
        optimizer.step()

        loss += train_prediction_loss.item()
        train_MAE+=MAE_temp
        train_r2+=r2_temp
        
    loss = loss / len(training_generator)
    train_MAE = train_MAE / len(training_generator)
    train_r2 = train_r2 / len(training_generator)

    return loss,train_MAE,train_r2,label_all

def evaluate(model, iterator,tes):
    model.eval()
    embedding_all = []
    pred = []
    with torch.no_grad():
        valid_MAE = 0
        valid_r2 = 0 
        for batch, label in iterator:
            # Model computations
            prediction = model(batch)
            MAE_temp = np.mean(np.abs(prediction.view(-1).detach().numpy()-label.detach().numpy()))
            r2_temp = r2_score(label.detach().numpy(),prediction.view(-1).detach().numpy())
            valid_MAE+=MAE_temp
            valid_r2+=r2_temp
            pred.append(prediction)
        valid_MAE = valid_MAE / len(iterator)
        valid_r2 = valid_r2 / len(iterator)

    return valid_MAE,valid_r2,pred

start = time.time()
r2_fcnn_all = []
MAE_fcnn_all = []
RMSE_fcnn_all = []
MAPE_fcnn_all = []

for t in range(5):
    print('Repeat RUN',t)
    splits = 5
    train_split_index,test_split_index = Kfold(n_sample,splits)
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    in_fea = 691
    h1 = 512
    h2 = 256
    latent_size = 128
    d_output = 1
    drop_rate = 0.5
    learning_rate = [0.001]#,0.0001]#,0.0005
    max_epochs = 1500
    batch_size = 128
    embeddings_all = []
    prediction_all = []
    r_fcnn = []
    
    # Generators
    for k in range(splits):
        print('current split is',k)
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

        training_set = HC50Dataset(np.arange(len(train_id)),train_feature,train_label)
        training_generator = DataLoader(training_set, batch_size=batch_size,shuffle=True, num_workers=0)

        validation_set = HC50Dataset(np.arange(len(valid_id)),valid_feature,valid_label)
        validation_generator = DataLoader(validation_set, batch_size=batch_size,shuffle=True, num_workers=0)

        test_set = HC50Dataset(np.arange(len(test_id)),test_feature,test_label)
        test_generator = DataLoader(test_set,  batch_size=batch_size,shuffle=False, num_workers=0)

        best_r2 = 0
        for l_r in learning_rate:
            print('learning rate is', l_r)
            model = FCNN(in_fea,h1,h2,latent_size,d_output,drop_rate)
            model = model.double()
            optimizer = optim.Adam(model.parameters(), lr=l_r,weight_decay=7e-5)
            criterion2 = nn.MSELoss()
            # Loop over epochs
            for epoch in range(max_epochs):
                train_loss,train_MAE,train_r2,train_label = train(model, training_generator, optimizer,criterion2)
                if epoch%200 == 0:
                    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, train_loss))
                    print('    train MAE is ',train_MAE)
                    print('    train r 2 is',train_r2)
                # Validation
                    if epoch>=200:
                        valid_MAE,valid_r2,_ = evaluate(model, validation_generator,False)
                        print('valid results MAE r2',valid_MAE,valid_r2)

                        if valid_r2 > best_r2:
                            best_r2 = valid_r2
                            test_MAE,test_r2,pred = evaluate(model, test_generator,True)
                            print('test MAE is ',test_MAE)
                            print('test r 2 is',test_r2)
        prediction_all.append(pred)
        r_fcnn.append(test_r2)
        print('best test result from this split is', test_r2)

    pred_fcnn_all = []
    for l in prediction_all:
        for v in l:
            for vv in v.detach().numpy():
                pred_fcnn_all.append(vv)
    prediction_true_fcnn_all = [HC50[i] for i in total_id]

    MAE = np.mean(np.abs(np.array(prediction_true_fcnn_all)-np.array(pred_fcnn_all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_fcnn_all), np.array(pred_fcnn_all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_fcnn_all), np.array(pred_fcnn_all).reshape(-1))

    r2_fcnn_all.append(np.mean(r_fcnn))
    MAE_fcnn_all.append(MAE)
    RMSE_fcnn_all.append(RMSE)
    MAPE_fcnn_all.append(MAPE)
end = time.time()

print('time is',(end-start)/60)

d1={'rsquared':str(np.mean(r2_fcnn_all).round(3))+u"\u00B1"+str(np.std(r2_fcnn_all).round(3)),
    'MAE':str(np.mean(MAE_fcnn_all).round(3))+u"\u00B1"+str(np.std(MAE_fcnn_all).round(3)),
    'RMSE':str(np.mean(RMSE_fcnn_all).round(3))+u"\u00B1"+str(np.std(RMSE_fcnn_all).round(3))
   }
OUTPUT= pd.DataFrame([d1])
print('FCNN result is',OUTPUT)


# In[ ]:




