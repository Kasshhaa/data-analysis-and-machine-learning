import json
import requests
import datetime

# 1.1 Extract Information From Transaction

# fetching data from the whatsonchain API
my_answer = requests.get("https://api.whatsonchain.com/v1/bsv/main/block/height/758992")
answer_json = json.loads(my_answer.text)

# obtaining the information from the block
print("txcount: " + str(answer_json["txcount"]))
print("time: " + str(answer_json["time"]))
print("totalFees: " + str(answer_json["totalFees"]))
print("confirmations: " + str(answer_json["confirmations"]))
print("miner: " + str(answer_json["miner"]))

# converting the unix timestamp into human read- able format to the nearest second.
timestamp = datetime.datetime.fromtimestamp(answer_json["time"])
print ("The block was mined on: " + str(timestamp.strftime("%Y-%m-%d %H:%M:%S")))


#2.1 Time Series Data

import pandas as pd
from fredapi import Fred
import numpy as np
fred = Fred(api_key="6f278b4aa22e45ccdd5f1abc83e61964")

import yfinance as yf
df_yahoo = yf.download('AAPL',
start='2014-12-01',
end='2022-11-01')

# series of the price of Coinbase Bitcoin
CBBTCUSD= fred.get_series("CBBTCUSD", title= "Coinbase Bitcoin", observation_start='2014-12-01',
                          observation_end='2022-11-01')

# series of the price of S&P 500
SP500 =fred.get_series("SP500",title= "S&P 500", observation_start='2014-12-01',
                          observation_end='2022-11-01')

#2.2 Data Transformations


# three data series  placed together into a Pandas DataFrame with compatible time periods
df = {}
df['CBBTCUSD'] = fred.get_series('CBBTCUSD', observation_start='2014-12-01', observation_end='2022-10-31')
df['APPLE'] = df_yahoo["Close"]
df['SP500'] = fred.get_series('SP500', observation_start='2014-12-01', observation_end='2022-10-31')
df = pd.DataFrame(df)
df

# Transforming observations into returns
new_df = np.log(df) - np.log(df.shift(1))
new_df.fillna(0)

#2.3 Data Data Analysis
import matplotlib.pyplot as plt
import numpy as np

# Make the plot
x = new_df.std().tolist()
y = new_df.mean().tolist()
symbols = new_df.columns
# Scatterplot and annotation
plt.scatter(x, y)
for index, symbol in enumerate(symbols):
    plt.annotate(symbol, (x[index], y[index]))

plt.xlabel('Risk')
plt.ylabel('Expected Return')
plt.title('Expected Return versus Risk')


# Add a constant variable
new_df['const'] = 1

reg1 = sma.OLS(endog=new_df["CBBTCUSD"], exog=new_df[['const',"SP500"]], missing='drop')
result= reg1.fit()
print(result.summary())

#3.3 Machine Learning Model
#Predicting if tomorrows price of apple stock would increase based on todays price

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

# reading in the csv file for apple
aapl = pd.read_csv('AAPL.csv', index_col = 'Date')
aapl.head()
aapl.plot.line(y="Close", use_index=True)


dataset = aapl[['Close']]
dataset = dataset.rename(columns = {'Close':'Test_Close'})

# assigning a score of 0 when the price went down and 1 when prices goes up
dataset["Score"] = aapl.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

# shifting the prices by one day
aapl_prev = aapl.copy()
aapl_prev = aapl_prev.shift(1)

data_columns = ["Close", "Volume", "Open", "High", "Low"]
dataset = dataset.join(aapl_prev[data_columns]).iloc[1:]
dataset = dataset.iloc[5000:]

# classification algorithm using the last 100 items to train and testing it on the rest of the data
model = RandomForestClassifier(n_estimators=1000, min_samples_split=50, random_state=1)

train = dataset.iloc[:-100]
test = dataset.iloc[-100:]

model.fit(train[data_columns], train["Score"])
RandomForestClassifier(min_samples_split=50, n_estimators=1000, random_state=1)


# making predictions across the entire dataset and train the model every 150 rows starting with 300 as initial set
def backtest(dataset, model, columns, start=300, step=150):
    predictions = []
    for i in range(start, dataset.shape[0], step):
        train = dataset.iloc[0:i].copy()
        test = dataset.iloc[i:(i+step)].copy()

        model.fit(train[data_columns], train["Score"])

# if model predicts a 60% chance that price would go up then it would issue 1 and 0 otherwise
        predicts = model.predict_proba(test[columns])[:,1]
        predicts = pd.Series(predicts, index=test.index)
        predicts[predicts > .6] = 1
        predicts[predicts<=.6] = 0

        combined = pd.concat({"Score": test["Score"],"Predictions": predicts}, axis=1)

        predictions.append(combined)

    return pd.concat(predictions)


predictions = backtest(dataset, model, data_columns)

# gives model accuracy
precision_score(predictions["Score"], predictions["Predictions"])
