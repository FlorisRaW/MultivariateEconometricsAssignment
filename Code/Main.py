import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import ADF
from arch.unitroot import DFGLS
from arch.unitroot import PhillipsPerron
from arch.unitroot import ZivotAndrews
from arch.unitroot import VarianceRatio
from arch.unitroot import KPSS
import csv

##### Multivariate Econometrics Assignment

### Global settings
workingDirectory = "C:/Users/User/OneDrive/Documents/University/Multivariate Econometrics/Assignment/Code/MultivariateEconometricsAssignment/"
exportPlots = True
exportFolderFigures = "./Outputs/Figures/"
exportFolderData = "./Outputs/Data/"
showFigures = False

# Setup
os.chdir(workingDirectory)
os.makedirs(exportFolderFigures, exist_ok=True)
os.makedirs(exportFolderData, exist_ok=True)

### Read data
data = pd.read_csv("./Data/MVE_assignment_2020_dataset.csv")
headers = {'cntry.name':'Country', 'year':'Year', 'mean_pre':'Precipitation', 'mean_rad':'Radiation', 'mean_tmp':'Temperature', 'NY.GDP.MKTP.KD':'Gdp', 'NY.GDP.PCAP.KD':'GdpPerCapita', 'SP.POP.TOTL':'Population', 'AG.LND.AGRI.K2':'AgriculturalLand', 'AG.PRD.CROP.XD':'CropProductionIndex'}
data.rename(columns=headers, inplace=True)

columnDescriptions = {'Country':'Country' , 'Year':'Year', 'Precipitation':'Mean precipitation', 'Radiation':'Mean radiation', 'Temperature':'Average yearly temperature', 'Gdp':'GDP (constant 2010 US$)', 'GdpPerCapita':'GDP per capita (constant 2010 US$)', 'Population':'Population, total', 'AgriculturalLand':'Agricultural land (sq. km)', 'CropProductionIndex':'Crop production index'}

### Question 1 - Plotting the data
print("### QUESTION 1 ###")

## Data cleaning functions
def CleanData(data):
    #the following columns are not needed
    data = data.drop(['ISO_N3', 'ISO_C3'], axis=1)

    #fix missing data and convert to correct data type
    data['AgriculturalLand'] = data['AgriculturalLand'].replace('..', np.nan)
    data['CropProductionIndex'] = data['CropProductionIndex'].replace('..', np.nan) 
    data['AgriculturalLand'] = pd.to_numeric(data['AgriculturalLand'])
    data['CropProductionIndex'] = pd.to_numeric(data['CropProductionIndex'])
    #print(data.dtypes)
    return data

## Data transformation functions
def TransformData(data):
    dataLogs = CalculateLogs(data)
    dataCombined = pd.concat([data, dataLogs], axis = 1)

    dataFirstDifferences = CalculateFirstDifferences(dataCombined)
    dataPercentageChange = CalculatePercentageChange(dataCombined)
    dataCombined2 =  pd.concat([dataCombined, dataFirstDifferences, dataPercentageChange], axis = 1)
    
    return dataCombined2

    #dataShifted1 = CalculateShiftByX(dataCombined1, -1)
    #dataCombined3 = pd.concat([dataCombined2, dataShifted], axis = 1)
    #return dataCombined3

def CalculateLogs(data):
    dataTransformed = data.apply(func=np.log, axis=0)
    dataTransformed = dataTransformed.drop('Year', axis=1)
    dataTransformed.columns = dataTransformed.columns + 'Log'
    return dataTransformed

def CalculateFirstDifferences(data):
    dataTransformed = data.diff()
    dataTransformed = dataTransformed.drop('Year', axis=1)
    dataTransformed.columns = dataTransformed.columns + 'DeltaAbs'
    return dataTransformed

def CalculatePercentageChange(data):
    dataTransformed = data.pct_change()
    dataTransformed = dataTransformed.drop('Year', axis=1)
    dataTransformed.columns = dataTransformed.columns + 'DeltaRel'
    return dataTransformed

def CalculateShiftByX(data, shift):
    dataTransformed = data.shift(shift)
    dataTransformed = dataTransformed.drop('Year', axis=1)
    dataTransformed.columns = dataTransformed.columns + 'Shift' + shift
    return dataTransformed

## Graph functions
def CreatePlots(data, country, columnDescriptions, showFigures=True, export=False, exportFolder=''):
    CreateCorrPlot(data, country + ' Correlation Plot', showFigures, exportPlots, exportFolder)
    CreateLinePlots(data, country, columnDescriptions, showFigures, exportPlots, exportFolder)

# Creates a correlation plot of all columns in dataframe
def CreateCorrPlot(data, title, showFigure=True, export=False, exportFolder=''):
    corr = data.corr()
    ax = sn.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sn.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.title(title)
    if export:
        plt.savefig(exportFolder + 'Q1_' + title + '.png', bbox_inches='tight')
    if showFigure:
        plt.show()
    else:
        plt.clf()

# Creates a simple x y plot
def CreateLinePlot(x, y, xLabel, yLabel, title, showFigure=True, export=False, exportFolder=''):
    plt.plot(x, y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if export:
        plt.savefig(exportFolder + 'Q1_' + title + '.png', bbox_inches='tight')
    if showFigure:
        plt.show()
    else:
        plt.clf()

# Creates a plot for each column in a dataframe (except country, year) with the column 'Year' as the x-axis
def CreateLinePlots(data, country, columnDescription, showFigures=True, export=False, exportFolder=''):
    for column in data.columns:
        if column not in ['Country', 'Year']:
            if column in columnDescriptions:
                columnDescription = columnDescriptions[column]
            else:
                columnDescription = column
            CreateLinePlot(data['Year'], data[column], 'Year', columnDescription, country + ' - ' + column, showFigures, export, exportFolder)

## Tests
def WriteTestResultsToCsv(results, outputFilePath):
    trendTypes = {'nc':'No trend components', 'c':'Include a constant','ct':'Include a constant and linear time trend','ctt':'Include a constant and linear and quadratic time trends'}
    with open(outputFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        #write column headers
        writer.writerow(['TestMethod', 'Column', 'TrendMethod','Trend','Lags','NullHypothesis','PValue','TestStatistic'])
        for testMethod in results:
            for column in results[testMethod]:
                for trendMethod in results[testMethod][column]:
                    for lag in results[testMethod][column][trendMethod]:
                        result = results[testMethod][column][trendMethod][lag]
                        try:
                            outTestMethod = testMethod
                            outColumn = column
                            outTrendMethod = trendMethod
                            outputTrend = trendTypes[trendMethod]
                            outLags = str(result.lags)
                            outNullHypothesis = result.null_hypothesis
                            outPValue = str(result.pvalue)
                            outStat = str(result.stat)
                            
                            writer.writerow([outTestMethod, outColumn, outTrendMethod, outputTrend, outLags, outNullHypothesis, outPValue, outStat])
                        except:
                            pass

def RunTests(data, printResults=False, writeResults=True, outputFilePath=''):
    results = dict()
    results["Adf"] = AugmentedDickeyFullerTest(data, printResults)
    results["Dfgls"] = DickeyFullerGlsTest(data, printResults)
    results["PhilipPerron"] = PhilipsPerronTest(data, printResults)
    results["ZivotAndrews"] = ZivotAndrewsTest(data, printResults)
    results["VarianceRatio"] = VarianceRatioTest(data, printResults)
    results["Kpss"] = KpssTest(data, printResults)

    if writeResults:
        WriteTestResultsToCsv(results,outputFilePath)

#Augmented Dickey Fuller test
#NOTE: Null hypthosesis is that there is a unit root
def AugmentedDickeyFullerTest(data, printResults=True, trend=None, lags=None):
    options_Trend = trend if trend != None else {'nc','c','ct','ctt'}
    options_Lags = lags if lags != None else {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}
    #options_LagMethod = lagMethod if lagMethod != None else {'AIC', 'BIC', 't-stat', None}

    results = dict()
    for column in data.columns:
        print("Augmented Dickey Fuller test for column: " + column)
        results_Trend = dict()
        for option_Trend in options_Trend:
            results_Lag = dict()
            for option_Lag in options_Lags:
                result = ADF(data[column].dropna(), trend=option_Trend, lags=option_Lag)
                if printResults:
                    result.summary()
                results_Lag[option_Lag] = result
            results_Trend[option_Trend] = results_Lag
        results[column] = results_Trend
    return results

#Dickey Fuller Gls Test
#NOTE: Null hypthosesis is that there is a unit root                      
def DickeyFullerGlsTest(data, printResults=True, trend=None, lags=None):
    options_Trend = trend if trend != None else {'c','ct'}
    options_Lags = lags if lags != None else {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}
    #options_LagMethod = lagMethod if lagMethod != None else {'AIC', 'BIC', 't-stat', None}

    results = dict()
    for column in data.columns:
        print("Dickey Fuller GLS test for column: " + column)
        results_Trend = dict()
        for option_Trend in options_Trend:
            results_Lag = dict()
            for option_Lag in options_Lags:
                result = DFGLS(data[column].dropna(), trend=option_Trend, lags=option_Lag)
                if printResults:
                    result.summary()
                results_Lag[option_Lag] = result
            results_Trend[option_Trend] = results_Lag
        results[column] = results_Trend
    return results

#Philips Perron Test
#NOTE: Null hypthosesis is that there is a unit root    
def PhilipsPerronTest(data, printResults=True, trend=None, lags=None):
    options_Trend = trend if trend != None else {'nc','c','ct'}
    options_Lags = lags if lags != None else {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}

    results = dict()
    for column in data.columns:
        print("Philips Perron test for column: " + column)
        results_Trend = dict()
        for option_Trend in options_Trend:
            results_Lag = dict()
            for option_Lag in options_Lags:
                result = PhillipsPerron(data[column].dropna(), trend=option_Trend, lags=option_Lag)
                if printResults:
                    result.summary()
                results_Lag[option_Lag] = result
            results_Trend[option_Trend] = results_Lag
        results[column] = results_Trend
    return results

#Zivot Andrews test
#NOTE: Null hypthosesis is that there is a unit root with a single structural break
def ZivotAndrewsTest(data, printResults=True, trend=None, lags=None):
    options_Trend = trend if trend != None else {'c','t','ct'} #{'nc','c','ct','ctt'}
    options_Lags = lags if lags != None else {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}
    #options_LagMethod = lagMethod if lagMethod != None else {'AIC', 'BIC', 't-stat', None}

    results = dict()
    for column in data.columns:
        print("Zivot Andrews test for column: " + column)
        results_Trend = dict()
        for option_Trend in options_Trend:
            results_Lag = dict()
            for option_Lag in options_Lags:
                result = ZivotAndrews(data[column].dropna(), trend=option_Trend, lags=option_Lag)
                if printResults:
                    result.summary()
                results_Lag[option_Lag] = result
            results_Trend[option_Trend] = results_Lag
        results[column] = results_Trend
    return results

#Variance Ratio Test
#NOTE: Null hypthosesis is that the process is a random walk, possibly plus drift. Rejection of the null with a positive test statistic indicates the presence of positive serial correlation in the time series
def VarianceRatioTest(data, printResults=True, trend=None, lags=None):
    options_Trend = trend if trend != None else {'nc','c'}
    options_Lags = lags if lags != None else {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}

    results = dict()
    for column in data.columns:
        print("Variance Ratio test for column: " + column)
        results_Trend = dict()
        for option_Trend in options_Trend:
            results_Lag = dict()
            for option_Lag in options_Lags:
                result = VarianceRatio(data[column].dropna(), trend=option_Trend, lags=option_Lag)
                if printResults:
                    result.summary()
                results_Lag[option_Lag] = result
            results_Trend[option_Trend] = results_Lag
        results[column] = results_Trend
    return results

#Kwiatkowski, Phillips, Schmidt and Shin (KPSS) Test
#NOTE: Null hypthosesis is that the series is weakly stationary and the alternative is that it is non-stationary. If the p-value is above a critical size, then the null cannot be rejected that there and the series appears stationary.
def KpssTest(data, printResults=True, trend=None, lags=None):
    options_Trend = trend if trend != None else {'c','ct'}
    options_Lags = lags if lags != None else {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}

    results = dict()
    for column in data.columns:
        print("Kwiatkowski, Phillips, Schmidt and Shin (KPSS) test for column: " + column)
        results_Trend = dict()
        for option_Trend in options_Trend:
            results_Lag = dict()
            for option_Lag in options_Lags:
                result = KPSS(data[column].dropna(), trend=option_Trend, lags=option_Lag)
                if printResults:
                    result.summary()
                results_Lag[option_Lag] = result
            results_Trend[option_Trend] = results_Lag
        results[column] = results_Trend
    return results

## START OF CODE
#Create outputs per country
for country in data['Country'].unique():
    print('### Creating results for country: ' + country)
    dataCountry = data[data['Country'] == country].drop('Country', axis=1)

    #Clean data
    dataCountry = CleanData(dataCountry)

    #Transform data
    dataCountry = TransformData(dataCountry)

    #Create outputs
    CreatePlots(dataCountry, country, columnDescriptions, showFigures, exportPlots, exportFolderFigures)
    
    #Run unit root tests
    RunTests(dataCountry.drop('Year', axis=1), outputFilePath=exportFolderData + 'TestResults_' + country + '.csv')