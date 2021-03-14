import pandas as pd
from sklearn.model_selection import train_test_split
import pickle as pkl
import numpy as np
import random

def categorical_to_varibles(dataset):
    cat_vars=['lifestyle','family_status','car','living_area','sports']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(dataset[var], prefix=var)
        data1=dataset.join(cat_list)
        dataset=data1
    return dataset

def to_keep_info(dataset):
    cat_vars=['lifestyle','family_status','car','living_area','sports','name','zip code','earnings']
    data_vars=dataset.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]
    final_dataset = dataset[to_keep]
    return final_dataset

def getdata(final_dataset):
    #get data
    X = final_dataset.loc[:, final_dataset.columns != 'label']
    y = final_dataset.loc[:, final_dataset.columns == 'label']

    return X,y

if __name__=='__main__':

    path="Recruiting_Task_InputData.csv"
    dataset=pd.read_csv(path)
    
    input_ = categorical_to_varibles(dataset)
    input_ = to_keep_info(input_)

    inputs_, labels_ = getdata(input_)
    
    for i in range(500):
        random_var = random.randint(0,10000)
        
        input_ = np.array(inputs_.loc[random_var]).reshape(-1,14)
        label_ = np.array(labels_.loc[random_var])
        #print(input_,label_)
        loaded_model = pkl.load(open('models/Logistic_regression.pkl', 'rb'))
        predicted  = loaded_model.predict(input_)
        print(predicted,label_)
