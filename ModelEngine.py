import pandas as pd
import numpy as np
from DataManager import *
from sklearn import neighbors
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from IndicatorGallexy import *
from sklearn import tree
from sklearn.linear_model import LogisticRegression

class ModelEngine:

    def __init__(self,chosen_model,train_data,test_data,val_ratio):
        self.origin_feature= train_data
        self.test_origin_data = test_data
        self.label =  pd.DataFrame()
        self.train_feature = pd.DataFrame()
        self.train_label = pd.DataFrame()
        self.val_feature = pd.DataFrame()
        self.val_label = pd.DataFrame()
        self.val_ratio = val_ratio
        self.test_feature = pd.DataFrame()
        self.onehot_cols = ['hour']
        self.normalize_cols = ['price_change']
        if chosen_model == 'Bayes':
            self.model = linear_model.BayesianRidge()
        elif chosen_model == 'MLPRegression':
            self.model = MLPRegressor()
        elif chosen_model == 'SVM':
            self.model = svm.SVR()
        elif chosen_model == 'LinearRegression':
            self.model = linear_model.LinearRegression()
        elif chosen_model == 'KNN':
            self.model = neighbors.KNeighborsClassifier()
        elif chosen_model == 'DT':
            self.model = tree.DecisionTreeClassifier()
        elif chosen_model == 'LR':
            self.model = LogisticRegression()
        else:
            print('Sorry,this method does not exist')


    # normalize and onehot
    def onehot(self):
        self.origin_feature = pd.get_dummies(self.origin_feature)

    def normalized_minmax(self):
        self.origin_feature=self.origin_feature.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    # for train and validation
    def setLabel(self,y_series):
        self.label = pd.DataFrame(y_series)
        return self.label

    def addFeatureTrain(self,name,x_series):
        self.origin_feature.insert(0,name,x_series)

    def delFeatureTrain(self,origin_feature,name):
        self.origin_feature.drop(name)

    def setFeature(self,dataframe):
        self.label=dataframe['label']
        dataframe=dataframe.drop('label',axis=1)
        self.origin_feature = dataframe

    def divide_train_val(self):
        self.origin_feature = self.origin_feature.dropna(axis=0,how='any')
        self.label = self.label.dropna(axis=0,how='any')
        data_num = len(self.origin_feature)
        val_num = int(data_num * self.val_ratio)
        self.train_feature = self.origin_feature[:-val_num]
        self.val_feature = self.origin_feature[-val_num:].reset_index(drop=True)
        self.train_label = self.label[:-val_num]
        self.val_label = self.label[-val_num:].reset_index(drop=True)

    # for test
    def addFeatureTest(self,name,x_series):
        self.test_feature.insert(0,name,x_series)

    def delFeatureTest(self,name):
        self.test_feature.drop(name)

    # train and predict
    def fit(self):
        self.model.fit(self.train_feature,self.train_label)

    def predictTest(self):
        predict_label = self.model.predict(self.test_feature)
        return predict_label

    def predictVal(self):
        self.predict_label = self.model.predict(self.val_feature)

    def score(self,predict_label):
        score = self.model.score(self.val_feature,predict_label)
        return score

    def r2_score(self,predict_label):
        score = self.model.r2_score(self.val_feature,predict_label)

    def getTrainFeature(self):
        return self.train_feature

    def getValFeature(self):
        return self.val_feature

    def getTestFeature(self):
        return self.test_feature

    def getTrainLabel(self):
        return self.train_label

    def getValLabel(self):
        return self.val_label

    def getPredictLabel(self):
        return self.predict_label

if __name__ == '__main__':
    DM = DataManager('D:\data\min')
    DM.loadSingleCSV('000010')
    DM.cleanData()
    train_data = DM.getDataFrame('000010')
    test_data = []
    modelEngine = ModelEngine('Bayes',train_data,test_data,0.2)
    #print(modelEngine.setLabel(train_data['p_change']))
    #modelEngine.setLabel(train_data['p_change'])
    train_data = DM.getAheadData('000010','p_change', 11)
    print(train_data)
    modelEngine.setFeature(train_data)
    # modelEngine.onehot()
    # modelEngine.normalized_minmax()
    # modelEngine.divide_train_val()
    # print(modelEngine.getTrainFeature())
    # print(modelEngine.getTrainLabel())
    # print(modelEngine.getValFeature())
    # print(modelEngine.getValLabel())
    # modelEngine.fit()
    # modelEngine.predictVal()
    # print(modelEngine.getPredictLabel())
    # print(modelEngine.score(modelEngine.getPredictLabel()))