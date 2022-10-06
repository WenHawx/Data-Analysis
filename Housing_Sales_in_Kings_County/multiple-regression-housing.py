import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# dataset obtained from: https://www.kaggle.com/harlfoxem/housesalesprediction?select=kc_house_data.csv

house_DF = pd.read_csv('kc_house_data.csv')
house_DF = house_DF.loc[house_DF['bedrooms'] <= 11]
housePrice = house_DF.iloc[:, 2].values.reshape(-1,1)
numBedrooms = house_DF.iloc[:, 3].values.reshape(-1,1)
sqFt_living = house_DF.iloc[:, 5].values.reshape(-1,1)

# Chart for correlation of price with number of bedrooms #########################################################
pearsonR, pValue = pearsonr(house_DF['price'], house_DF['bedrooms'])
pValue *= 2 
print(f"pValue for correlation of price with number of bedrooms: {pValue:.3f}")

linReg = LinearRegression()
linReg.fit(numBedrooms, housePrice)

yHat = linReg.predict(numBedrooms)
plt.scatter(numBedrooms, housePrice)
plt.plot(numBedrooms, yHat, color='red')
plt.title("Price of Home vs Number of Bedrooms")
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price of House (USD) e6')
plt.savefig('Price_VS_NumBedrooms.png')
################################################################################################################
plt.clf()


# Chart for correlation of price with square footage of living space ##################################################
pearsonR, pValue = pearsonr(house_DF['price'], house_DF['sqft_living'])
pValue *= 2 
print(f"pValue for correlation of price with square footage of living space: {pValue:.3f}")

linReg = LinearRegression()
linReg.fit(sqFt_living, housePrice)

yHat = linReg.predict(sqFt_living)
plt.scatter(sqFt_living, housePrice)
plt.plot(sqFt_living, yHat, color='red')
plt.title("Price of Home vs Square footage of living space")
plt.xlabel('Square footage of living space')
plt.ylabel('Price of House (USD) e6')
plt.savefig('Price_VS_sqft-living.png')
######################################################################################################################

# This section is splitting the data from the csv where 75 percent of the data is randomly selected to train a model to predict housing prices
# It then uses the other 25% to test the regression model to generate a r2 score
# It takes 30 runs and compiles a list of each run's r2 score and it's standard deviation

house_DF2 = pd.read_csv('kc_house_data.csv')
house_DF2 = house_DF2.loc[house_DF2['bedrooms'] <= 11]
house_DF2 = house_DF2.drop('id', axis=1)
house_DF2 = house_DF2.drop('date', axis=1)
house_DF2 = house_DF2.drop('lat', axis=1)
house_DF2 = house_DF2.drop('long', axis=1)

x = house_DF2.drop('price', axis=1)
y = house_DF2['price']

numRuns = 1
r2ScoreList = []

while numRuns <= 30:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, shuffle=True)

    print(x_train.head())
    print(y_train.head())
    linReg = LinearRegression()
    linReg.fit(x_train, y_train)
    y_prediction = linReg.predict(x_test)
    r2Score = r2_score(y_test, y_prediction)
    r2ScoreList.append(r2Score)
    numRuns+=1

print("\nR2 score statistics")
print("---------------------")
print(f"Mean: {np.mean(r2ScoreList):.3f}")
print(f"Standard Deviation: {np.std(r2ScoreList):.3f}")
