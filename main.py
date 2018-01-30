import argparse
from DataManager import DataManager
from ModelEngine import ModelEngine
from IndicatorGallexy import IndicatorGallexy
from DrawResult import *
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("trainpath", type = str, help = "the directory of the train data")
    parser.add_argument("-t","--testpath", type = str, default='none',help = "the directory of the test data")
    parser.add_argument("-f","--features", type = str , default = 'p_change',help = "choose the features")
    parser.add_argument("-l","--label", type = str, default='p_change',help = 'choose the label')
    parser.add_argument("code", type = str, help = "the code of stock to be predicted")
    parser.add_argument("-m","--model", type = str, default = 'MLPRegression',help="choose the training model")
    parser.add_argument("-i","--aheadinterval",type = int, default=11)
    parser.add_argument("-r","--valratio", type = int, default=0.2,help="the ratio of the validtion")
    parser.add_argument("-c","--corrstock",type = bool, default=0, help="use correlative stock or not")
    args = parser.parse_args()

    folderpath = args.trainpath
    code = args.code
    model = args.model
    interval = args.aheadinterval
    feature = args.features
    label = args.label
    ratio = args.valratio
    interval = args.aheadinterval


    '''
        get data and clean
    '''
    dataManager = DataManager(folderpath)
    dataManager.loadFolderDataDict()
    dataManager.cleanData()
    train_data = dataManager.getDataDict()[code]

    if args.corrstock == 1:
        train_data = dataManager.addCorrData(code,'p_change')

    if not args.testpath=='none':
        dataManagerTest = DataManager(args.t)
        dataManagerTest.loadFolderDataDict()
        dataManagerTest.cleanData()
        test_data = dataManagerTest.getDataDict()[code]
    else:
        test_data = []



    '''
        model engine
    '''
    modelEngine = ModelEngine(model, train_data, test_data, ratio)
    if feature == 'p_change':
        # use the ahead p_change as the feature
        train_set = dataManager.getAheadData(code,'p_change', interval)
        #print(train_set)
        modelEngine.setFeature(train_set)
    else:
        modelEngine.setLabel(train_data[label])
        features = feature.split(',')
        for fea in features:
            modelEngine.addFeatureTrain(train_data[fea],fea)
    #modelEngine.onehot()
    modelEngine.divide_train_val()
    # print(modelEngine.getTrainFeature())
    # print(modelEngine.getTrainLabel())
    # print(modelEngine.getValFeature())
    # print(modelEngine.getValLabel())
    modelEngine.fit()
    modelEngine.predictVal()
    predict_label = modelEngine.getPredictLabel()
    score = modelEngine.score(modelEngine.getPredictLabel())


    '''
        draw the picture of the result
    '''
    if not os.path.exists('..\ForecastFrame\pic'):
        os.mkdir('..\ForecastFrame\pic')
    drawResult = DrawResult()
    drawResult.DivergingPic(modelEngine.getValLabel(),predict_label)
    IG = IndicatorGallexy(dataManager.getDataFrame(code),dataManager.getDataDict())
    corr_matrix = IG.getCorrMatrix('p_change')
    drawResult.drawCorr(corr_matrix)
    drawResult.drawFeature(train_data['close'])