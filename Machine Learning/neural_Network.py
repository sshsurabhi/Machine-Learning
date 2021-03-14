import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

class Trainer():
    def __init__(self,path="Recruiting_Task_InputData.csv"):
        super().__init__()
        #self.dataset_path= path
        self.dataset=pd.read_csv(path)
        self.final_dataset=None
        self.cat_vars= ['lifestyle','family_status','car','living_area','sports']
        self.to_keep_vars=['lifestyle','family_status','car','living_area','sports','name','zip_code']
        self.model = None
        self.X_train =None
        self.X_test=None
        self.y_train=None
        self.y_test =None
        self.model=None
        
    def categorical_to_varibles(self):
        cat_vars=self.cat_vars
        #cat_vars=['lifestyle','family_status','car','living_area','sports']
        for var in cat_vars:
            cat_list='var'+'_'+var
            cat_list = pd.get_dummies(self.dataset[var], prefix=var)
            data1=self.dataset.join(cat_list)
            self.dataset=data1

    def to_keep_info(self):
            cat_vars=self.to_keep_vars
            #cat_vars=['lifestyle','family_status','car','living_area','sports','name','zip code','earnings']
            data_vars=self.dataset.columns.values.tolist()
            to_keep=[i for i in data_vars if i not in cat_vars]
            self.final_dataset = self.dataset[to_keep]

    def getdata(self):
        ##### get data
        X = self.final_dataset.loc[:, self.final_dataset.columns != 'label']
        y = self.final_dataset.loc[:, self.final_dataset.columns == 'label']
        #print(X, X.shape)
        ##### encode to integers
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        ##### train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, encoded_Y, test_size=0.3)
    
    def create_baseline(self):
        # create model
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=14, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.50))
        self.model.add(Dense(1, activation='sigmoid'))
        # Compile model
        Adam1 = keras.optimizers.Adam(lr=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer= Adam1, metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train,epochs=50, batch_size=32, verbose=1)
        #plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['val', 'test'], loc='upper right')
        plt.show()

    def loss_evaluate(self):
        y_pred = self.model.predict(self.X_test)
        log_loss(self.y_test, y_pred)

    def train(self):
        ##### preprocess
        self.categorical_to_varibles()
        self.to_keep_info()
        self.getdata()
        self.create_baseline()
        self.loss_evaluate()
        

if __name__=='__main__':
    clf = Trainer()
    clf.train()
    



