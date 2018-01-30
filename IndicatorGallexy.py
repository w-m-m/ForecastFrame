import pandas as pd
import numpy as np
from DataManager import *
from DataManager import DataManager


class IndicatorGallexy:

    def __init__(self,data_df,data_dict):
        self.data_df = data_df
        self.data_dict = data_dict
        self.indicator_dict={}
        pass

    def loadIndicator(self,name,indicator_df):
        if name not in self.indicator_dict.keys():
            self.indicator_dict[name] = indicator_df
        else:
            pd.concat([self.indicator_dict[name]],indicator_df)

    def calLog(self,minuend,meiosis):
        # log(minuend) - log(meiosis)
        minuend_price = self.data_df[minuend]
        meiosis_price = self.data_df[meiosis]
        res = []
        for (minp,meip) in zip(minuend_price,meiosis_price):
            temp = np.math.log(minp) - np.math.log(meip)
            res.append(temp)
        name = 'log_'+meiosis+'_'+minuend
        indicator = pd.DataFrame(res,columns=[name])
        self.loadIndicator(name,indicator)

        a = np.log(self.data_df[minuend]) - np.log(self.data_df[meiosis])

    def calEMA(self,shortNumber,longNumber,attriname):
        """
            计算移动平均值，快速移动平均线为12日，慢速移动平均线为26日
            快速：EMA[i] = EMA[i-1] * (short - 1)/(short + 1) + close * 2 / (short + 1)
            慢速：EMA[i] = EMA[i-1] * (long - 1)/(long + 1) + close * 2 / (long + 1)
        """
        ema_short = [self.data_df[attriname][0]] * len(self.data_df)
        ema_long = [self.data_df[attriname][0]] * len(self.data_df)
        for i in range(1,len(self.data_df)):
            ema_short[i] = ema_short[i-1] * (shortNumber-1)/(shortNumber+1) + self.data_df[attriname][i] * 2/(shortNumber+1)
            ema_long[i] = ema_long[i-1] * (longNumber-1)/(longNumber+1) + self.data_df[attriname][i] *2/(longNumber+1)
        ema_short_name = 'ema'+str(shortNumber)
        ema_short = pd.DataFrame(ema_short,columns=[ema_short_name])
        ema_long_name = 'ema'+str(longNumber)
        ema_long = pd.DataFrame(ema_long,columns=[ema_long_name])
        self.loadIndicator(ema_short_name,ema_short)
        self.loadIndicator(ema_long_name,ema_long)

    def calDIF(self,emashortname,emalongname):
        """
            DIF为离差值，涨势中，离差值会变得越来越大，跌势中，离差值会变得越来越小
            DIF = EMA(short) - EMA(long)
        """
        dif = self.indicator_dict[emashortname] - self.indicator_dict[emalongname]
        dif_name = 'dif'
        dif = pd.DataFrame(dif,columns=[dif_name])
        self.loadIndicator(dif_name,dif)

    def calDEA(self,difname,n):
        """
            计算DEA差离平均值
            DEA[i] = DEA[i-1] * (n-1) / (n+1) + DIF[i] * 2 / (n+1)
            其中n为多少日
        """
        dea = [self.indicator_dict[difname][0]]*len(self.data_df)
        for i in range(1,len(self.data_df)):
            dea[i] = dea[i-1]*(n-1)/(n+1) + self.indicator_dict[difname][i] * 2/( n+1)
        dea_name = 'dea'
        dea = pd.DataFrame(dea,columns=[dea_name])
        self.loadIndicator(dea_name,dea)

    def calMACD(self,difname,deaname):
        """
            计算MACD指数平滑移动平均线
            MACD = 2 * (DIF - DEA)
        """
        macd = 2*(self.indicator_dict[difname] - self.indicator_dict[deaname])
        macd_name = 'macd'
        macd = pd.DataFrame(macd,columns=[macd_name])
        self.loadIndicator(macd_name,macd)

    def calBOLL(self,indc_name,days_num):
        """
            布林线指标，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
            中轨线 = N日的移动平均线
            上轨线 = 中轨线 + 两倍的标准差
            下轨线 = 中轨线 - 两倍的标准差
            策略：股价高于区间，卖出；股价低于，买入
        """
        ma_days = pd.DataFrame()
        ma_days = pd.rolling_mean(self.data_df[indc_name],days_num)
        temp = [0]*len(self.data_df)
        for i in range(19,len(self.data_df)):
            temp[i] = self.data_df[indc_name][max(i-(days_num-1),0):i+1].std()
        data_std_name = 'ma_std'
        data_std = pd.DataFrame(temp,columns=[data_std_name])
        self.loadIndicator(data_std_name,data_std)

        data_bollname = 'midboll'
        data_boll = pd.DataFrame(ma_days,columns=[data_bollname])
        self.loadIndicator(data_bollname,data_boll)

        data_upboll_name = 'upboll'
        data_upboll = pd.DataFrame(data_boll+2*data_std , columns=[data_upboll_name])
        self.loadIndicator(data_upboll_name,data_upboll)

        data_lowboll_name = 'lowboll'
        data_lowboll = pd.DataFrame(data_boll-2*data_std)
        self.loadIndicator(data_lowboll_name,data_lowboll)


    def getCorrMatrix(self,feature):
        corr_pairs = pd.DataFrame()
        for key in self.data_dict.keys():
            corr_pairs[key] = self.data_dict[key][feature]
        corr = corr_pairs.corr()
        corr_df = pd.DataFrame(corr,index = self.data_dict.keys(),columns=self.data_dict.keys())
        corr_df = corr_df.fillna(0)
        return corr_df

    def getCorrCode(self,feature,code):
        corr = self.getCorrMatrix(feature)
        corr_code = set()
        corr = corr[corr.abs() >= 0.3]
        corr = corr.dropna(axis=1, how='all').dropna(axis=0, how='all')
        for index in corr.loc[code].index:
            #print(index)
            if abs(corr.loc[code][index]) > 0:
                corr_code.add(index)
        return corr_code

    def getCol(self,column):
        return self.data_df[column]

    def calAvg(self,column):
        return self.data_df[column].mean()

    def calMin(self,column):
        return self.data_df[column].min()

    def calMax(self,column):
        return self.data_df[column].max()

    def calVar(self,column):
        # 方差
        return self.data_df[column].var()

    def calStd(self,column):
        #标准差
        return self.data_df[column].std()

    def calMad(self,column):
        # 平均绝对离差
        return self.data_df[column].mad()

    def calPriceChange(self,open,p_change):
        return float(open*p_change)

    def calClose(self,open,p_change):
        return float(open * (1+p_change))

    def calMA(self):
        # 移动均值
        sum = self.data_df['close'].apply(lambda x:x.sum(),axis=1)
        MA = float(sum/len(self.data_df['close']))
        return MA


if __name__ == '__main__':
    DM = DataManager('D:\data\min')
    DM.loadFolderDataDict()
    data_dict = DM.getDataDict()
    DM.cleanData()
    indicatorGallexy = IndicatorGallexy(data_dict['000010'],data_dict)
    corr = indicatorGallexy.getCorrMatrix('p_change')
    #print(corr)
    code_set = indicatorGallexy.getCorrCode('p_change','000010')
    #print(code_set)