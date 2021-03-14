
############importing needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle as pkl
import os 
import argparse

########## create models directory for saving the model/weights file
if not os.path.exists('models'):
    os.makedirs('models')

########## given dataset can be analysed by the following steps
def data_analysis(verbose):
    if verbose:
        dataframe=pd.read_csv("Recruiting_Task_InputData.csv")
        count_no_res = len(dataframe[dataframe['label']== "no response"])
        print("\n no response",count_no_res)
        count_res = len(dataframe[dataframe['label']=="response"])
        print("\n response",count_res)
        pct_of_no_res = count_no_res/(count_no_res+count_res)
        print("\n percentage of no response is", pct_of_no_res*100)
        pct_of_res = count_res/(count_no_res+count_res)
        print("\n percentage of response", pct_of_res*100)

        #### plotting lbel column for predicting "response" and "no reponse" people
        sns.countplot(x='label', data = dataframe, palette='hls')
        
        #### predict percentage of "response" and "no response" people from label column
        print(dataframe.groupby("label").mean())

        #### predict no of different category of people those are involed for advertisement
        print(dataframe["family_status"].value_counts())

        #### predict percentage of people with different lifestyle
        print(dataframe.groupby("lifestyle").mean())

        #### predict percentage of people with different cars
        print(dataframe.groupby("car").mean())

        #### predict percentage of people with different family status
        print(dataframe.groupby("family_status").mean())

        #### predict percentage of people living in different places those involved int dataset
        print(dataframe.groupby("living_area").mean())
        
        #### creating crosstab bar between life style column and label column
        pd.crosstab(dataframe.lifestyle,dataframe.label).plot(kind='bar')
        plt.title('response Frequency')
        plt.xlabel('Job')
        plt.ylabel('Frequency of Purchase')
        plt.show()
        #plt.savefig('purchase_fre_job')

        #### creating crosstab bar between family status column and label column 
        table=pd.crosstab(dataframe.family_status,dataframe.label)
        table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
        plt.title('Stacked Bar Chart of Marital Status vs Purchase')
        plt.xlabel('Marital Status')
        plt.ylabel('Proportion of Customers')
        plt.show()
        #plt.savefig('mariral_vs_pur_stack')

        #### creating crosstab bar between living area column and label column
        table=pd.crosstab(dataframe.living_area,dataframe.label)
        table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
        plt.title('Stacked Bar Chart of Education vs Purchase')
        plt.xlabel('Education')
        plt.ylabel('Proportion of Customers')
        plt.show()
        #plt.savefig('edu_vs_pur_stack')

        #### creating histogram for age column to analyze the distribution of age of the people.
        dataframe.age.hist()
        plt.title('Histogram of Age')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.show()
        #plt.savefig('hist_age')


fig=plt.figure
class Trainer():
    def __init__(self,path="Recruiting_Task_InputData.csv"):
        super().__init__()
        #self.dataset_path= path
        self.dataset=pd.read_csv(path)
        self.final_dataset=None
        self.cat_vars= ['lifestyle','family_status','car','living_area','sports']
        self.to_keep_vars=['lifestyle','family_status','car','living_area','sports','name','zip code','earnings']
        self.model = None
        self.X_train =None
        self.X_test=None
        self.y_train=None
        self.y_test =None
        self.tpr=None
        self.fpr=None
        self.auc=None


    ##### function that converts all string values columns in to one-hot coded values
    def categorical_to_varibles(self):
        cat_vars=self.cat_vars
        #cat_vars=['lifestyle','family_status','car','living_area','sports']
        for var in cat_vars:
            cat_list='var'+'_'+var
            cat_list = pd.get_dummies(self.dataset[var], prefix=var)
            data1=self.dataset.join(cat_list)
            self.dataset=data1

    #### function used to neglect some columns those are not needed after they converted into one-hot coded values
    #### function name is opposite way around.
    #### after this function we receives final_dataset after all the necessary columns were converted in to one-hot coding
    #### this final_dataset will be used for training and testing
    def to_keep_info(self):
        cat_vars=self.to_keep_vars
        #cat_vars=['lifestyle','family_status','car','living_area','sports','name','zip code','earnings']
        data_vars=self.dataset.columns.values.tolist()
        to_keep=[i for i in data_vars if i not in cat_vars]
        self.final_dataset = self.dataset[to_keep]

    #### saving a model file using pickle
    def save_model(self, name):
        pkl.dump(self.model, open(name, 'wb'))

    #### function used to split the final_dataset into 2 parts (70:30) ratio
    def getdata(self):
        #get data
        X = self.final_dataset.loc[:, self.final_dataset.columns != 'label']
        y = self.final_dataset.loc[:, self.final_dataset.columns == 'label']
        ##### train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)
        
    ##### converting string values of label column in to numeric format    
    def convert_to_numeric(self,labels):
        new_labels=[]
        for i in range(len(labels)):
            if labels[i]=='response':
                new_labels.append(1)
            else:
                new_labels.append(0)
        return new_labels
    
    ##### function to create roc curve
    def roc_curve(self, y_pred):
        y_pred = self.convert_to_numeric(y_pred)
        y_test = np.array(self.y_test).flatten()
        y_test = self.convert_to_numeric(y_test)
        #### compute fpr, tpr, thresholds and roc auc
        y_pred_proba = self.model.predict_proba(self.X_test)[::,1]
        self.fpr, self.tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        self.auc = metrics.roc_auc_score(y_test, y_pred_proba)

    def RandomForestCLF(self):
        ##### random forest training
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train.values.ravel())
        y_pred = self.model.predict(self.X_test)
        self.plot_evaluation(y_pred,'Random Forest')

    def LogisticRegressionCLF(self):
        #### Logistic Regression training
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train.values.ravel())
        y_pred = self.model.predict(self.X_test)
        self.plot_evaluation(y_pred,'Logistic Regression')

    def SVMclf(self):
        ##### SVM training
        self.model= svm.SVC(gamma=0.001, C=1,probability=True)
        self.model.fit(self.X_train, self.y_train.values.ravel())
        y_pred = self.model.predict(self.X_test)
        self.plot_evaluation(y_pred,'SVM')
        
    def KNNclf(self):
        ##### KNN
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X_train, self.y_train.values.ravel())
        y_pred = self.model.predict(self.X_test)
        self.plot_evaluation(y_pred,'KNN')
    
    def plot_evaluation(self, y_pred, method):
        print('\n Accuracy of '+method+' classifier on test set: {:.2f}'.format(self.model.score(self.X_test, self.y_test)))
        c_matrix = confusion_matrix(self.y_test, y_pred)
        print("\n"+method+" c_matrix", c_matrix)
        scores = cross_val_score(self.model, self.X_train, self.y_train.values.ravel(), cv=5)
        print('\n '+method+ ' Cross-Validation Accuracy Scores', scores)
        self.save_model('models/'+method +'.pkl')
        self.roc_curve(y_pred)
        plt.plot(self.fpr, self.tpr)
        plt.title(method +" accuracy="+str(self.auc))
        plt.show()

    def Train(self, clf):
        ##### preprocess
        self.categorical_to_varibles()
        self.to_keep_info()
        self.getdata()
        if clf=='Random':
            self.RandomForestCLF()
        elif clf=='Logistic':
            self.LogisticRegressionCLF()
        elif clf =='SVM':
            self.SVMclf()
        elif clf == 'KNN':
            self.KNNclf()
        else:
            self.RandomForestCLF()
            self.LogisticRegressionCLF()
            self.SVMclf()
            self.KNNclf()
    plt.show()
    #plt.savefig('Graphs.png')

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("approach",help="type approach: KNN,Random,SVM,Logistic, or all",default='all')
    args = parser.parse_args()
    
    data_analysis(False)
    cl = Trainer()
    cl.Train(args.approach)




    

