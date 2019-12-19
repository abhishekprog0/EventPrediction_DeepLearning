# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
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
    
    lastIndex = beginIndex+1
    for i in range(beginIndex, n):
        if (data_dates[i] > stoppingDate):
            lastIndex = i
            break
        
    X_train = X.iloc[range(beginIndex,lastIndex)]
    Y_train = Y.iloc[range(beginIndex,lastIndex)]
    X_test = X.iloc[range(lastIndex, lastIndex+5)]
    Y_test = Y.iloc[range(lastIndex, lastIndex+5)]
        
    return (X_train, Y_train), (X_test, Y_test)

def createPortfolio1(predDf, dataset, datasetDates, long_only = False, exclude_citi = False):
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
            
            if (foundIndex >= len(df)):
                foundIndex = len(df) - 1
            
            closePrice = df.iloc[foundIndex]['Close']
            
            
            proportions[j-1] = eps / closePrice
            if (long_only and proportions[j-1] < 0):
                proportions[j-1] = 0
            if (exclude_citi and ((j-1) == ticker.index('Citigroup'))):
                proportions[j-1] = 0
        
        if (long_only and sum(proportions) == 0):
            proportions[:] = 1
        proportions = proportions / sum(proportions)
        weightDict = {}
        weightDict['Date'] = currDate
        for j in range(1,k):
            weightDict[weightColumns[j]] = proportions[j-1]
        
        weightDf = weightDf.append(weightDict, ignore_index = True)
        
    return weightDf

def createPortfolio2(predDf, dataset, datasetDates, long_only = False, exclude_citi = False):
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
            if (foundIndex >= len(df)):
                foundIndex = len(df) - 1
            
            analyst = df.iloc[foundIndex].get('EPS_Analyst')
            if (analyst == None):
                proportions[j-1] = 0
            else:
                proportions[j-1] = (eps - analyst) / eps

            if (long_only and proportions[j-1] < 0):
                proportions[j-1] = 0
            if (exclude_citi and ((j-1) == ticker.index('Citigroup'))):
                proportions[j-1] = 0
        
        if (long_only and sum(proportions) == 0):
            proportions[:] = 1

        proportions = proportions / sum(proportions)
        weightDict = {}
        weightDict['Date'] = currDate
        for j in range(1,k):
            weightDict[weightColumns[j]] = proportions[j-1]
        
        weightDf = weightDf.append(weightDict, ignore_index = True)
        
    return weightDf

def createPortfolio3(predDf, dataset, datasetDates, long_only = False, exclude_citi = False):
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

            if (foundIndex >= len(df)):
                foundIndex = len(df) - 1

            startDateToConsider = datetime(currDate.year-1, currDate.month, 1)  
            for jj in range(len(dfDates)):
                if (dfDates[jj] > startDateToConsider):
                    startDateIndex = jj
                    break
            
            analyst = df.iloc[foundIndex].get('EPS_Analyst')
            if (analyst == None):
                proportions[j-1] = 0
            else:
                proportions[j-1] = (eps - analyst) / analyst

            if (long_only and proportions[j-1] < 0):
                proportions[j-1] = 0
            if (exclude_citi and ((j-1) == ticker.index('Citigroup'))):
                proportions[j-1] = 0
        
        if (long_only and sum(proportions) == 0):
            proportions[:] = 1
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

            if (startIndex >= len(df)):
                startIndex = len(df) - 1
            if (endIndex >= len(df)):
                endIndex = len(df) - 1
                
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

def plotAgainstSP(prices, weight, spPrices, title = '', imageFile = ''):
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
    totalData['SP_Index'] = spCleanPrices
    totalData['Date'] = weight.Date
    plt.figure()
    for j in range(m):
        plt.plot( totalData.Date, totalData[j])
    plt.plot(totalData.Date, totalData.SP_Index)
    plt.title(title)
    plt.legend()
    plt.savefig(imageFile)
    
    print(title)
    print(ir)

    return relatives, absolutes, trackingErrors, alphas, ir, returns, vols, sharpes, spCleanPrices
    
# MAIN code starts here
long_only_values = [True, False]
exclude_citi_values = [True, False]
financials_only = [True, False]
combos = list(itertools.product(long_only_values, exclude_citi_values, financials_only))

for c in range(len(combos)):
    print('*** Beginning Loop ***')
    
    combo = combos[c]
    long_only = combo[0]
    exclude_citi = combo[1]
    financials = combo[2]
    print('Long Only: ' + str(long_only))
    print('Exclude Citi: ' + str(exclude_citi))
    print('Financials: ' + str(financials))


    if (financials):    
        ticker = ['WellsFargo', 'GoldmanSachs', 'BankOfAmerica', 'BerkshireHathaway', 'Blackrock', 'BNYMellon', 'Citigroup', 'JPMorgan', 'MorganStanley']
    else:
        ticker = ['WellsFargo', 'GoldmanSachs', 'BankOfAmerica', 'BerkshireHathaway', 'Blackrock', 'BNYMellon', 'Citigroup', 'JPMorgan', 'MorganStanley','Adobe', 'Apple', 'NVIDIA']
    
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
    
    if (financials):
        portfolioStopDate = datetime(2018,1,1)
    else:
        portfolioStopDate = datetime(2019,11,1)

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
    
    doPortfolio1 = False
    doPortfolio2 = False
    doPortfolio3 = False
    
    print('*** Beginning Construction Phase ***')
    
    combo = combos[c]
    long_only = combo[0]
    exclude_citi = combo[1]

    weight1File = 'weights1_' + str(long_only) + '_' + str(exclude_citi) + '_' + str(financials) + '.xlsx'
    weight2File = 'weights2_' + str(long_only) + '_' + str(exclude_citi) + '_' + str(financials) + '.xlsx'
    weight3File = 'weights3_' + str(long_only) + '_' + str(exclude_citi) + '_' + str(financials) + '.xlsx'
    imageFile = 'plot_' + str(long_only) + '_' + str(exclude_citi) + '_' + str(financials) + '.png'

    if (doPortfolio1):
        weight1Df = createPortfolio1(predDf, dataset, datasetDates, long_only = long_only, exclude_citi = exclude_citi)
        weight1Df.to_excel(weight1File)
    weight1Df = pd.read_excel(weight1File)
        
    if (doPortfolio2):
        weight2Df = createPortfolio2(predDf, dataset, datasetDates, long_only = long_only, exclude_citi = exclude_citi)
        weight2Df.to_excel(weight2File)
    weight2Df = pd.read_excel(weight2File)
        
    if (doPortfolio3):
        weight3Df = createPortfolio3(predDf, dataset, datasetDates, long_only = long_only, exclude_citi = exclude_citi)
        weight3Df.to_excel(weight3File)
    weight3Df = pd.read_excel(weight3File)
    
    weights = [weight1Df, weight2Df, weight3Df]  

    print('*** Beginning Return Calc Phase ***')
    
    doPrices = False
    pricesFile = 'prices_' + str(long_only) + '_' + str(exclude_citi) + '_' + str(financials) + '.xlsx'
    if (doPrices):
        prices = calcReturns(weights, dataset, datasetDates)
        pd.DataFrame(prices).to_excel(pricesFile)
    prices = pd.read_excel(pricesFile)
    
    print('*** Beginning Analytics Phase ***')
    
    if (financials):
        spPrices = pd.read_excel('SPF prices.xls')
    else:
        spPrices = pd.read_excel('SPX prices.xls')
    
    title = ('Financials ' if financials else 'All Sectors ') + ' - '
    title = title + ('Long Only' if long_only else 'Long Short')
    title = title + ' - '
    title = title + ('No Citi' if exclude_citi else 'With Citi')
    
    relatives, absolutes, trackingErrors, alphas, ir, returns, vols, sharpes, spCleanPrices = \
        plotAgainstSP(prices, weights[0], spPrices, title = title, imageFile = imageFile)

