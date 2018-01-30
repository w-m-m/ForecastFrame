import  pandas as pd
import numpy as np
from IndicatorGallexy import *
import os

from IndicatorGallexy import IndicatorGallexy


class DataManager:

    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.data_dict={}

    def loadFolderDataDict(self):
        # load all data of a folder to data_dict
        for root,dirs,files in os.walk(self.folder_path):
            for file in files:
                file = file.split('.')
                code = file[0]
                code_df = self.loadSingleCSV(code)
                if code not in self.data_dict.keys():
                    self.data_dict[code] = code_df

    def loadFromWebsite(self):
        pass

    def loadChosenDataDict(self,codes):
        # load the input codes' csv to data_dict
        codes = codes.split(',')
        for code in codes:
            # print(code)
            code_df = self.loadSingleCSV(code)
            if code not in self.data_dict.keys():
                self.data_dict[code] = code_df

    def loadSingleCSV(self,code):
        # load dataframe from code.csv
        csv_path = code+'.csv'
        single_df = pd.read_csv(self.folder_path+'\\'+csv_path)
        # print(single_df)
        if code not in self.data_dict.keys():
            self.data_dict[code] = single_df

    def getDataDict(self):
        return self.data_dict

    def getDataFrame(self,code):
        # return dataframe from data_dict according to the key  'code'
        return self.data_dict[code]

    def splitData(self,code):
        # a = pd.to_time( df.date)
        # df['min'] = a.min
        # df['sec'] = a.sec
        # a.year
        #


        # split date to year month day hour min sec
        date_split = pd.DataFrame((x.split(' ') for x in self.data_dict[code].date), columns=['day', 'time'])
        time_split = pd.DataFrame((x.split(':') for x in date_split.time), columns=['hour', 'min', 'sec'])
        time_split.drop('sec', axis=1)
        day_split = pd.DataFrame((x.split('-') for x in date_split.day), columns=['year', 'month', 'day'])
        self.data_dict[code].drop('date', axis=1)
        self.data_dict[code] = pd.concat([day_split, time_split, self.data_dict[code]], axis=1)

    def cleanData(self):
        # clean missing-value data and nan
        for code in self.data_dict.keys():
            self.data_dict[code].dropna()
            self.splitData(code)

    def dumpToHDF5(self):
        store = pd.HDFStore(os.path.join(self.folder_path,'data.h5'))
        for item in self.data_dict.items():
            code = item[0]
            df = item[1]
            store.put(code,df)
        store.close()

    def addCorrData(self,code,feature):
        data_df = self.data_dict[code]
        IG = IndicatorGallexy(self.data_dict[code],self.data_dict)
        corr_code = IG.getCorrCode(feature,code)
        for code in corr_code:
            data_df.append(self.data_dict[code])
        return data_df


    # def getAheadData(self,column,interval):
    #     code_df = self.data_df
    #     data_set = []
    #     for index in range(len(code_df)-interval):
    #         row = code_df.iloc[index]
    #         day = row['day']
    #         row_ahead = code_df[index+interval]
    #         day_ahead = row_ahead['day']
    #         if day_ahead==day:
    #             data = list(code_df[index+1:interval+index+1][column])
    #             data.append(row['hour'])
    #             data.append(row['p_change'])
    #             data_set.append(data)
    #     data_set = pd.DataFrame(data_set)
    #     return data_set

    def getAheadData(self,code,column,interval = 11):
        data_set = pd.DataFrame()
        data_df = self.data_dict[code]
        for i in range(1,interval+1):
            data_set["p_change_%d" % i] = data_df[column].shift(-i)
        data_set['hour']=data_df['hour']
        data_set['label']=data_df[column]
        data_set = data_set.dropna(axis=0,how='any')
        return data_set

if __name__ == '__main__':
    DM = DataManager('D:\data\min')
    # DM.loadSingleCSV('000010')
    # DM.cleanData()
    # DM.loadChosenDataDict('000005,000008,000010,000020')
    # DM.loadFolderDataDict()
    # DM.cleanData(DM.getDataDict())
    print (DM.getDataDict())