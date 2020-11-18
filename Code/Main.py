import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

##### Multivariate Econometrics Assignment

### Global settings
exportPlots = True
exportFolder = "./Outputs/Figures/"
showFigures = False

# Setup
os.makedirs(exportFolder, exist_ok=True)

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
    data.drop(['ISO_N3', 'ISO_C3'], axis=1, inplace=True)

    #fix missing data and convert to correct data type
    data['AgriculturalLand']= data['AgriculturalLand'].replace('..', np.nan)
    data['CropProductionIndex']= data['CropProductionIndex'].replace('..', np.nan) 
    data['AgriculturalLand'] = pd.to_numeric(data['AgriculturalLand'])
    data['CropProductionIndex'] = pd.to_numeric(data['CropProductionIndex'])
    print(data.dtypes)

## Data transformation functions
def AddFirstDifferencesColumns(data):
    dataFirstDifferences = data.diff()
    dataFirstDifferences.drop('Year', axis=1, inplace=True)
    dataFirstDifferences.columns = dataFirstDifferences.columns + 'Delta'
    return pd.concat([data, dataFirstDifferences], axis=1)

## Graph functions
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
    plt.savefig(exportFolder + 'Q1_' + title + '.png')
    if showFigure:
        plt.show()
    else:
        plt.clf()

# Creates a simple x y plot
def CreatePlot(x, y, xLabel, yLabel, title, showFigure=True, export=False, exportFolder=''):
    plt.plot(x, y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.savefig(exportFolder + 'Q1_' + title + '.png')
    if showFigure:
        plt.show()
    else:
        plt.clf()

# Creates a plot for each column in a dataframe (except country, year) with the column 'Year' as the x-axis
def CreatePlots(data, country, columnDescription, showFigures=True, export=False, exportFolder=''):
    for column in data.columns:
        if column not in ['Country', 'Year']:
            if column in columnDescriptions:
                columnDescription = columnDescriptions[column]
            else:
                columnDescription = column
            CreatePlot(data['Year'], data[column], 'Year', columnDescription, country + ' - ' + column, showFigures, export, exportFolder)

## START OF CODE
# Clean data
CleanData(data)

#Create outputs per country
for country in data['Country'].unique():
    dataCountry = data[data['Country'] == country].drop('Country', axis=1)
    dataCountry = AddFirstDifferencesColumns(dataCountry)
    #Create outputs
    CreateCorrPlot(dataCountry, country + ' Correlation Plot', showFigures, exportPlots, exportFolder)
    CreatePlots(dataCountry, country, columnDescriptions, showFigures, exportPlots, exportFolder)
