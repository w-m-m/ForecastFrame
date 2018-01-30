import numpy as np
import seaborn as srn
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *

class DrawResult(object):

    def __init__(self):
        pass

    # draw the comparation of the predicted_result and the actual_value
    def DivergingPic(self,test_label,predict_lable):
        num = range(len(test_label))
        sns.set()
        plt.plot(num,test_label, label=u'test_lablel')
        plt.plot(num,predict_lable , label=u'predict_label')
        plt.legend()
        #plt.show()
        plt.savefig('..\ForecastFrame\pic\DiverginPic.png')

    # draw the correlation matrix of the stocks
    def drawCorr(self,corr_matrix):
        srn.heatmap(corr_matrix)
        plt.show()
        #plt.savefig('..\ForecastFrame\pic\Correlation.png')

    # draw the value of a feature
    def drawFeature(self,series):
        sns.distplot(series)
        plt.show()
        #plt.savefig('..\ForecastFrame\pic\Feature.png')