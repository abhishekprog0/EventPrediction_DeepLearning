# -*- coding: utf-8 -*-


import pandas as pd
import quandl
import numpy as np
import quandl
import matplotlib.pyplot as plt
import os
import math
#import talib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from sklearn.metrics import accuracy_score
import seaborn as sns
import operator
import warnings
warnings.filterwarnings("ignore")

plt.rc('figure', figsize=(20, 8), dpi=100)
from datetime import datetime

class Plot:
    
    def __init__(self, data):
        self.data = data
        
    def autocorr(self, col, ticker):
        
        plot_acf(self.data[col], title = col + ' Autocorrelation Plot of ' + ticker)
        plt.show()
            
    def featureImportance(self, feature_importance, num_features, flag, ticker):
    
        f = dict()
        n = len(feature_importance)
        for i in range (n):
            f[X_test.columns[i]] = feature_importance[i]
        f = sorted(f.items(), key=operator.itemgetter(1), reverse=True)
        f = f[:num_features]
        feature_name = list()
        feature_values = list()
        for i, j in f:
            feature_name.append(i)
            feature_values.append(j)
        fig = plt.figure(figsize=(14,5))
        plt.xticks(rotation='vertical')
        plt.bar([i for i in range(len(f))], feature_values, tick_label=feature_name)
        if flag == 1:
            plt.title('Feature importance for EPS Prediction of ' + ticker + ' (Excluding Analyst estimate features)')
        else:
            plt.title('Feature importance for EPS Prediction of ' + ticker + ' (Including Analyst estimate features)')
        plt.show()
    
    def lossStatsAndCurve(self, X_test, Y_test, regressor, ticker):
        
        
        rmse = np.sqrt(mean_squared_error(Y_test, regressor.predict(X_test)))
        print("Root Mean Squared Error: %f" % (rmse))
        #print ("Regression Prediction Score: " + str(round(regressor.score(X_test,Y_test) * 100, 2)) + "%")
        eval_result = regressor.evals_result()
        training_rounds = range(len(eval_result['validation_0']['rmse']))
        plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
        plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Training Vs Validation Error of ' + ticker)
        plt.legend()
        plt.show()

class Model:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
    
    def trainModel(self, epoch = 1000, verbose_flag = False, learning_rate = 0.01):
        
        regressor = xgb.XGBRegressor(colsample_bytree = 0.4, learning_rate = learning_rate, base_score=0.65, max_depth = 4, alpha = 10, n_estimators = epoch)
        xgbModel=regressor.fit(X_train, Y_train,eval_set = [(X_train, Y_train), (X_test, Y_test)], verbose = verbose_flag)
        return (xgbModel, regressor)

def getData(data_temp, inc_analyst):
    
    data = data_temp.copy()
    Y = data['EPS (diluted)']
    
    del data['EPS (recurring)']
    del data['EPS (diluted)']
    
    if inc_analyst == False:
        #del data['Growth (YoY%)_Analyst']
        del data['EPS_Analyst']
    X = data
    
    train_samples = int(X.shape[0] * 0.75)
     
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    Y_train = Y.iloc[:train_samples]
    Y_test = Y.iloc[train_samples:]
    
    
    return (X_train, Y_train), (X_test, Y_test)

def getDataUntilDate(data_temp, data_dates, startingDate, stoppingDate):
    data = data_temp.copy()
    Y = data['EPS (diluted)']
    
    del data['EPS (recurring)']
    del data['EPS (diluted)']
    
    X = data
    n = X.shape[0]
    for i in range(n):
        if (data_dates[i] > startingDate):
            beginIndex = i
            break
        
    for i in range(beginIndex, n):
        if (data_dates[i] > stoppingDate):
            lastIndex = i
            break
        
    X_train = X.iloc[range(beginIndex,lastIndex)]
    Y_train = Y.iloc[range(beginIndex,lastIndex)]
    X_test = X.iloc[range(lastIndex, lastIndex+5)]
    Y_test = Y.iloc[range(lastIndex, lastIndex+5)]
        
    return (X_train, Y_train), (X_test, Y_test)

def createPortfolio1(predDf, dataset, datasetDates):
    n = predDf.shape[0]
    weightColumns = predDf.columns[1:]
    weightDf = pd.DataFrame(columns = weightColumns)
    k = len(weightColumns)

    for i in range(n):
        currRow = predDf.iloc[i]
        currDate = currRow['Date']
        proportions = np.zeros(k-1)
        for j in range(1,k):
            eps = currRow[weightColumns[j]]
            df = dataset[j-1]
            dfDates = datasetDates[j-1]
            for ii in range(len(dfDates)):
                if (dfDates[ii] > currDate):
                    foundIndex = ii
                    break
            
            closePrice = df.iloc[foundIndex]['Close']
            proportions[j-1] = eps / closePrice
        proportions = proportions / sum(proportions)
        weightDict = {}
        weightDict['Date'] = currDate
        for j in range(1,k):
            weightDict[weightColumns[j]] = proportions[j-1]
        
        weightDf = weightDf.append(weightDict, ignore_index = True)
        
    return weightDf

def createPortfolio2(predDf, dataset, datasetDates):
    epsilon = 0.0001
    n = predDf.shape[0]
    weightColumns = predDf.columns[1:]
    weightDf = pd.DataFrame(columns = weightColumns)
    k = len(weightColumns)

    for i in range(n):
        currRow = predDf.iloc[i]
        currDate = currRow['Date']
        proportions = np.zeros(k-1)
        for j in range(1,k):
            eps = currRow[weightColumns[j]]
            df = dataset[j-1]
            dfDates = datasetDates[j-1]
            for ii in range(len(dfDates)):
                if (dfDates[ii] > currDate):
                    foundIndex = ii
                    break
            
            analyst = df.iloc[foundIndex]['EPS_Analyst']
            propToUse = (eps - analyst) / eps
            if (propToUse < epsilon):
                propToUse = epsilon
            proportions[j-1] = propToUse
            

        proportions = proportions / sum(proportions)
        weightDict = {}
        weightDict['Date'] = currDate
        for j in range(1,k):
            weightDict[weightColumns[j]] = proportions[j-1]
        
        weightDf = weightDf.append(weightDict, ignore_index = True)
        
    return weightDf

def createPortfolio3(predDf, dataset, datasetDates):
    epsilon = 0.0001
    n = predDf.shape[0]
    weightColumns = predDf.columns[1:]
    weightDf = pd.DataFrame(columns = weightColumns)
    k = len(weightColumns)

    for i in range(n):
        currRow = predDf.iloc[i]
        currDate = currRow['Date']
        proportions = np.zeros(k-1)
        for j in range(1,k):
            eps = currRow[weightColumns[j]]
            df = dataset[j-1]
            dfDates = datasetDates[j-1]
            for ii in range(len(dfDates)):
                if (dfDates[ii] > currDate):
                    foundIndex = ii
                    break
            startDateToConsider = datetime(currDate.year-1, currDate.month, 1)  
            for jj in range(len(dfDates)):
                if (dfDates[jj] > startDateToConsider):
                    startDateIndex = jj
                    break
            
            analyst = np.average(df.iloc[startDateIndex:foundIndex]['EPS_Analyst'])
            propToUse = (eps - analyst) / analyst
            if (propToUse < epsilon):
                propToUse = epsilon
            proportions[j-1] = propToUse
        proportions = proportions / sum(proportions)
        weightDict = {}
        weightDict['Date'] = currDate
        for j in range(1,k):
            weightDict[weightColumns[j]] = proportions[j-1]
        
        weightDf = weightDf.append(weightDict, ignore_index = True)
        
    return weightDf

def calcReturns(weights, dataset, datasetDates):
    n = weights[0].shape[0]
    m = len(weights)
    weightColumns = weights[0].columns[2:]
    k = len(weightColumns)    

    prices = np.zeros([n,m])
    for j in range(m):
        prices[0,j] = 100
        
    for i in range(1,n):
        periodStartDate = weights[0].iloc[i-1].Date
        periodEndDate = weights[0].iloc[i].Date
        prch = np.zeros(k)
        
        for j in range(k):
            # find price change of asset j between periodStartDate and periodEndDate
            df = dataset[j]
            dfDates = datasetDates[j]
            for ii in range(len(dfDates)):
                if(dfDates[ii] >= periodStartDate):
                    startIndex = ii
                    break
            
            for ii in range(startIndex, len(dfDates)):
                if(dfDates[ii] >= periodEndDate):
                    endIndex = ii
                    break
                
            startPrice = df.iloc[startIndex]['Close']    
            endPrice = df.iloc[endIndex]['Close']
            prch[j] = (endPrice - startPrice) / startPrice
        
        for j in range(m):
            totalPrch = 0
            for jj in range(k):
                totalPrch = totalPrch + weights[j].iloc[i][weightColumns[jj]] * prch[jj]
            prevPrice = prices[i-1,j]
            nextPrice = prevPrice * (1+totalPrch)
            prices[i,j] = nextPrice
    
    return prices

def plotAgainstSP(prices, weight, spPrices):
    n = prices.shape[0]
    m = prices.shape[1]-1
    
    spCleanPrices = np.zeros(n)
    spCleanPrices[0] = 100
    
    relatives = np.zeros([n-1,m])
    absolutes = np.zeros([n-1,m])
    for i in range(1,n):
        endDate = weight.Date[i]
        startDate = weight.Date[i-1]
        for ii in range(len(spPrices.Date)):
            if(spPrices.Date[ii] >= startDate):
                startIndex = ii
                break
        for ii in range(startIndex,len(spPrices.Date)):
            if(spPrices.Date[ii] >= endDate):
                endIndex = ii
                break
        startSpPrice = spPrices.Price[startIndex]
        endSpPrice = spPrices.Price[endIndex]
        spPrch = (endSpPrice - startSpPrice) / startSpPrice
        spCleanPrices[i] = (spPrch + 1) * spCleanPrices[i-1]
        for j in range(m):
            start = prices.iloc[i-1][j]
            end = prices.iloc[i][j]
            prch = (end - start) / start
            relatives[i-1,j] = prch - spPrch
            absolutes[i-1,j] = prch
    
    # Calculate analytics
    trackingErrors = np.std(relatives,0)
    alphas = np.sum(relatives,0)
    ir = np.zeros(m)
    
    returns = np.sum(absolutes,0)
    vols = np.std(absolutes,0)
    sharpes = np.zeros(m)
    
    for j in range(m):
        ir[j] = alphas[j] / trackingErrors[j]
        sharpes[j] = returns[j] / vols[j]
    
    totalData = prices.copy()
    totalData['SP500'] = spCleanPrices
    totalData['Date'] = weight.Date
    for j in range(m):
        plt.plot( totalData.Date, totalData[j])
    plt.plot(totalData.Date, totalData.SP500)
    plt.legend()

    return relatives, absolutes, trackingErrors, alphas, ir, returns, vols, sharpes, spCleanPrices
    
# MAIN code starts here
ticker = ['WellsFargo', 'GoldmanSachs', 'BankOfAmerica', 'BerkshireHathaway', 'Blackrock', 'BNYMellon', 'Citigroup', 'JPMorgan', 'MorganStanley']

os.chdir("C:\\Users\\hdharmaw\\OneDrive - GMO\\Documents\\4742\\project\\EventPrediction_DeepLearning\\FinalData")

dataset = []
for i in range(len(ticker)):
    df = pd.read_excel(ticker[i] + '_Final.xlsx')
    print(ticker[i])
    print('Total dataset has {} days, and {} features.'.format(df.shape[0], df.shape[1]))
    dataset.append(df)
    
datasetDates = []

for item in dataset:
    datasetDates.append(item['Date'])
    del item['Date']


trainingStartDate = datetime(2004,1,1)
portfolioStartDate = datetime(2010,1,1)
portfolioStopDate = datetime(2018,1,1)

currDate = portfolioStartDate

predColumns = ['Date']
for i in range(len(ticker)):
    predColumns.append(ticker[i])
predDf = pd.DataFrame(columns = predColumns)

os.chdir("C:\\Users\\hdharmaw\\OneDrive - GMO\\Documents\\4742\\project\\EventPrediction_DeepLearning\\PortfolioConstruction")

doPrediction = False
predFile = 'predictions.xlsx'
print('*** Beginning Prediction Phase ***')
if (doPrediction):
    retrain = False
    regressorDict = {}
    while (currDate <= portfolioStopDate):
        print(currDate)
        currDateDict = {'Date': currDate}
    
        for i in range(len(dataset)):
            df = dataset[i]
            dfDates = datasetDates[i]

            (X_train, Y_train), (X_test, Y_test) = getDataUntilDate(df, dfDates, trainingStartDate, currDate)        
            
            if (retrain or (currDate == portfolioStartDate)):
                m1 = Model(X_train, Y_train, X_test, Y_test)
                xgbModel, regressor = m1.trainModel(verbose_flag=False)
                regressorDict[i] = regressor
            else:
                regressor = regressorDict[i]
                
            prediction = np.average(regressor.predict(X_test))
            currDateDict[predColumns[i+1]] = prediction
            
        predDf = predDf.append(currDateDict, ignore_index=True)
            
        if (currDate.month == 12):
            currDate = datetime(currDate.year+1, 1, 1)   
        else:
            currDate = datetime(currDate.year, currDate.month+1, 1)  
    
    predDf.to_excel(predFile)
    
predDf = pd.read_excel(predFile)

print('*** Beginning Construction Phase ***')

doPortfolio1 = False
doPortfolio2 = False
doPortfolio3 = False

weight1File = 'weights1.xlsx'
weight2File = 'weights2.xlsx'
weight3File = 'weights3.xlsx'

if (doPortfolio1):
    weight1Df = createPortfolio1(predDf, dataset, datasetDates)
    weight1Df.to_excel(weight1File)
weight1Df = pd.read_excel(weight1File)
    
if (doPortfolio2):
    weight2Df = createPortfolio2(predDf, dataset, datasetDates)
    weight2Df.to_excel(weight2File)
weight2Df = pd.read_excel(weight2File)
    
if (doPortfolio3):
    weight3Df = createPortfolio3(predDf, dataset, datasetDates)
    weight3Df.to_excel(weight3File)
weight3Df = pd.read_excel(weight3File)

weights = [weight1Df, weight2Df, weight3Df]  

print('*** Beginning Return Calc Phase ***')

doPrices = False
pricesFile = 'prices.xlsx'
if (doPrices):
    prices = calcReturns(weights, dataset, datasetDates)
    pd.DataFrame(prices).to_excel(pricesFile)
prices = pd.read_excel(pricesFile)

print('*** Beginning Analytics Phase ***')

spPrices = pd.read_excel('sp500.xls')
relatives, absolutes, trackingErrors, alphas, ir, returns, vols, sharpes, spCleanPrices = plotAgainstSP(prices, weights[0], spPrices)

#i = 0
#for df in dataset:
#    (X_train, Y_train), (X_test, Y_test) = getData(df, inc_analyst = True)
#
#    m1 = Model(X_train, Y_train ,X_test, Y_test)
#    xgbModel1, regressor1 = m1.trainModel(verbose_flag=False)
#
#    plot.lossStatsAndCurve(X_test, Y_test, regressor1, ticker[i])
#    feature_importance = xgbModel1.feature_importances_.tolist()
#    plot.featureImportance(feature_importance, 15, 0, ticker[i])
#    i = i + 1