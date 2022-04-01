#This is the code for different layers-7 layers
#691-512-256-128-256-512-691

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
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.decomposition import PCA
import seaborn as sns
import time

#data preprocessing
descriptors_data = pd.read_csv('HC50raw.csv')
descriptors_data_processed = descriptors_data.dropna(axis=1)
HC50 = descriptors_data_processed.iloc[:, 1]
HC50 = np.array(HC50)
descriptors = descriptors_data_processed.iloc[:, 2:]
descriptors = np.array(descriptors)
n_sample = len(descriptors)

#main code
#autoencoder function
#Note, 7 layers
class AutoEncoder(nn.Module):
    def __init__(self,in_fea,h1,h2,latent_size,d_output,drop_rate):
        super(AutoEncoder,self).__init__()
        
        self.l1 = nn.Linear(in_fea,h1)
        self.l2 = nn.Linear(h1,h2)
        self.l3 = nn.Linear(h2,latent_size)
        self.l4 = nn.Linear(latent_size,h2)
        self.l5 = nn.Linear(h2,h1)
        self.l6 = nn.Linear(h1,in_fea)
        
        self.l7 = nn.Linear(latent_size,d_output)
        self.drop = nn.Dropout(drop_rate)
        self.act1 = nn.Tanh()
        self.act2 = nn.Sigmoid()
    def forward(self,x):
        x1 = self.drop(F.relu(self.l1(x)))
        x2 = self.drop(F.relu(self.l2(x1)))
        embedding = F.relu(self.l3(x2))
        x1_ = F.relu(self.l4(embedding))
        x2_ = F.relu(self.l5(x1_))
        x3_ = self.l6(x2_)
        
        x4 = self.l7(embedding)
        return x3_,embedding,x4

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

def train(model, iterator, optimizer, criterion1,criterion2):
    
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

        rebuild,embedding,prediction = model(train_batch)
        embedding_all.append(embedding)
        label_all.append(train_label)
        
        
        train_reconstruction_loss = criterion1(rebuild, train_batch)
        train_prediction_loss = criterion2(prediction.view(-1), train_label)
        MAE_temp = np.mean(np.abs(prediction.view(-1).detach().numpy()-train_label.detach().numpy()))
        r2_temp = r2_score(train_label.detach().numpy(),prediction.view(-1).detach().numpy())

        train_loss = train_reconstruction_loss+train_prediction_loss
        train_loss.backward()
        optimizer.step()

        loss += train_reconstruction_loss.item()
        train_MAE+=MAE_temp
        train_r2+=r2_temp
        
    loss = loss / len(training_generator)
    train_MAE = train_MAE / len(training_generator)
    train_r2 = train_r2 / len(training_generator)

    return loss,train_MAE,train_r2,embedding_all,label_all

def evaluate(model, iterator):
    model.eval()
    embedding_all = []
    pred = []
    with torch.no_grad():
        valid_MAE = 0
        valid_r2 = 0 
        valid_rmse = 0
        valid_mape = 0
        for batch, label in iterator:
            # Model computations
            _,embedding,prediction = model(batch)
            MAE_temp = np.mean(np.abs(prediction.view(-1).detach().numpy()-label.detach().numpy()))
            rmse_temp = mean_squared_error(label.detach().numpy(), prediction.view(-1).detach().numpy(), squared=False)
            mape_temp = mean_absolute_percentage_error(label.detach().numpy(), prediction.view(-1).detach().numpy())
            r2_temp = r2_score(label.detach().numpy(),prediction.view(-1).detach().numpy())
            valid_MAE+=MAE_temp
            valid_rmse += rmse_temp
            valid_mape +=mape_temp
            valid_r2+=r2_temp
            embedding_all.append(embedding)
            pred.append(prediction)
        valid_MAE = valid_MAE / len(iterator)
        valid_r2 = valid_r2 / len(iterator)
        valid_rmse = valid_rmse/len(iterator)
        valid_mape =valid_mape/len(iterator)
    return valid_MAE,valid_rmse,valid_mape,valid_r2,embedding_all,pred

start = time.time()
MAE_all = []
RMSE_all = []
MAPE_all = []
Rsquared_all =[]
for i in range(5):
    print('Repeat run',i)
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    splits = 5
    train_split_index,test_split_index = Kfold(n_sample,splits)
    in_fea = 691
    h1 = 512
    h2 = 256
    latent_size = 128
    d_output = 1
    drop_rate = 0.5
    learning_rate = [0.001]
    max_epochs = 1000
    batch_size = 128
    embeddings_all = []
    prediction_all = []
    r_all = []
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
            model = AutoEncoder(in_fea,h1,h2,latent_size,d_output,drop_rate)
            model = model.double()
            optimizer = optim.Adam(model.parameters(), lr=l_r,weight_decay=7e-5)
            criterion1 = nn.MSELoss()
            criterion2 = nn.MSELoss()
            # Loop over epochs
            for epoch in range(max_epochs):
                train_loss,train_MAE,train_r2,train_embedding,train_label = train(model, training_generator, optimizer, criterion1,criterion2)
                if epoch%200 == 0:
                    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, train_loss))
                    print('    train MAE is ',train_MAE)
                    print('    train r 2 is',train_r2)
                # Validation
                    if epoch>=200:
                        valid_MAE,_,_,valid_r2,_,_ = evaluate(model, validation_generator)
                        print('valid results MAE r2',valid_MAE,valid_r2)

                        if valid_r2 > best_r2:
                            best_r2 = valid_r2
                            test_MAE,test_rmse,test_mape,test_r2,embeddings,pred = evaluate(model, test_generator)
                            print('test MAE is ',test_MAE)
                            print('test r 2 is',test_r2)
        embeddings_all.append(embeddings)
        prediction_all.append(pred)
        r_all.append(test_r2)
        print('best test result from this split is', test_r2)
    print('final result',np.mean(r_all))
    print('final result std',np.std(r_all))
    pred__all = []
    for l in prediction_all:
        for v in l:
            for vv in v.detach().numpy():
                pred__all.append(vv)
    prediction_true_all = [HC50[i] for i in total_id]
    MAE = np.mean(np.abs(np.array(prediction_true_all)-np.array(pred__all).reshape(-1)))
    RMSE = mean_squared_error(np.array(prediction_true_all), np.array(pred__all).reshape(-1), squared=False)
    MAPE = mean_absolute_percentage_error(np.array(prediction_true_all), np.array(pred__all).reshape(-1))
    
    Rsquared_all.append(np.mean(r_all))
    MAE_all.append(MAE)
    RMSE_all.append(RMSE)
    MAPE_all.append(MAPE)
    print('final MAE',MAE)
    print('final RMSE',RMSE)
end = time.time()

duration = (end-start)/5
print('one runtime in minutes',duration/60)
print ('full run time in minutes',(end-start)/60)

d1={'rsquared':str(np.mean(Rsquared_all).round(3))+u"\u00B1"+str(np.std(Rsquared_all).round(3)),
    'MAE':str(np.mean(MAE_all).round(3))+u"\u00B1"+str(np.std(MAE_all).round(3)),
    'RMSE':str(np.mean(RMSE_all).round(3))+u"\u00B1"+str(np.std(RMSE_all).round(3)),
   }
OUTPUT= pd.DataFrame([d1])
OUTPUT

OUTPUT.to_csv('691-512-256-128-256-512-691.csv',sep='\t')

