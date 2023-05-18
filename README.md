# data-analysis-and-machine-learning

1 Explore the Bitcoin Blockchain and Basic Web Coding

1.1 Extract Information From Your Own Transaction

At the start of the module you set up Handcash Wallets, and I sent everyone a micropayment. Use the payment you received to do the following (if you did not participate, please ask other students to help you):
Go to your transaction history and find a way to locate the transaction on the blockchain. All wallets have a feature for viewing the transaction on the blockchain. Take a note of which block your transaction is in by taking its block height.
From a Jupyter notebook, extract the following information from the same block by fetching data from the whatsonchain API.
https://api.whatsonchain.com/v1/bsv/main/block/height/ place block height here
Your notebook should fetch, then print your data in JSON format, and you should obtain the following for the block with your transaction in it:
– txcount
– time
– totalFees
– confirmations – miner
Include some code that converts the unix timestamp into human read- able format to the nearest second.
Explain what each of these parts of the block are in words.

2 Time Series Investigation of Bitcoin Price

You are working for a FinTech firm that provides customers with real time financial data and analysis. You are tasked with providing a blog with ex- ample analysis that extracts a live data set, does some analysis, and draws some conclusions.
2.1 Obtain Time Series Data 
Obtain the following data by calling the FRED api (or any other of your choice) from a Jupyter notebook, and provide simple time series plots of the raw data: You must provide plots of three time series
1. A chosen price of a cryptocurrency, or any other individual stock that is considered to be high risk
2. A chosen price of an asset that is considered safe, like a stock price for a well known large company (if you do not know how to obtain this from other free api services like yahoo or quandl, please use FRED’s Gold price index with code ID7108)
3. Anindexmeasureofoverallstockmarketperformance(egtheS&P500).
Be sure to label your three series clearly, so that anyone reading your code can easily understand the analysis.
2.2 Data Transformations 
Choose the longest possible time span to conduct your analysis.
Make sure that your 3 data series are placed together into a Pandas
DataFrame, with compatible time periods
Transform observations into returns by obtaining new series:
xt ! xt−1
where xt is the value of a variable for a particular observation and xt−1 is its value 1 time period before.

 
2.3 Data Analysis
What is the correlation between the returns on risky and safe assets, and the market returns?
Interpret these results with respect to CAPM theory.
According to the assumptions behind the strict form of CAPM theory, equations of the following form should fully explain returns to holding any particular asset, here for bitcoin as an example with subscript b, and sub- script m refers to the overall market (eg the S&P500 index returns).
rbt = αb + βbrmt + ubt
where ubt is an idiosyncratic unpredictable error term associated with Bitcoin. According to the strict form of CAPM, α should be zero, and β provides a systematic measure of how high up the risk/return trade-off the asset is. Estimate α and β for your chosen risky asset, using OLS regression, and interpret the results.

3 Machine Learning in Practice 
Please note that the Python modelling is contained in the ‘model’ folder. A recording of this session with full subtitles can also be found on ele under the ‘TOPIC 4 AI and Machine Learning for FinTech’.

3.2 Written Description of Python Code 
Reproduce Sarunas’ model (saved under model building.ipynb) within your own Jupyter notebook. To do this you will have to download the large dataset from Kaggle following Sarunas’ instructions. This data will need to be saved in your active Jupyter notebook directory. Once you have reproduced it with the same results, using cell markdown, choose 5 lines of the code and include brief verbal descriptions of what those lines perform. Finally, save this as your own Jupyter notebook and include this in your submission. Do not submit the dataset, only the code and results.
3.3 Build your own Machine Learning Model
Choose your own dataset and machine learning model to produce predictions. You may use the same Kaggle dataset as Sarunas used, or choose your own, and we suggest making use of one of the machine learning algorithms offered by the Sckit library. This section is open ended, for you to explore what you want but it must be within the realm of prediction using financial data.
