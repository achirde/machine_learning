import pandas as pd
import numpy  as np

from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# File Names 
TRAIN_TIMESERIES = 'input_csv_files/train_timeseries2.csv'
TRAIN_TWEETER    = 'input_csv_files/train_tweets.csv'
TEST_TWEETER    = 'input_csv_files/test_tweets.csv'

# Model Selection 
model = ['r', 'l', 'g', 's']

# GB Parameters
LOSS        = 'huber'
CRITERION   = 'friedman_mse'
MAX_DEPTH   = 32

# Configuration
ENABLE_TWEET = True

columns = ['close',
#           'blocks_size',
#           'cost_per_transaction',
#           'difficulty',
#           'est_transaction_volume_USD',
#           'miners_revenue',
#           'total_bitcoins',
#           'transaction_fees_USD',
#           'mempool_count_max',
#           'wallets_created',
           
           'cost_per_transaction',
           'est_transaction_volume_USD',
           'est_transaction_volume',
           'cost_per_transaction_percent' ,
           'transaction_fees',
           'n_transactions_per_block',
           'utxo_count_max' ,
           'utxo_count_min' ,
           'mempool_size_max',
           
            ]

columns = list(set(columns))



# Read Files
data = pd.read_csv(TRAIN_TIMESERIES, index_col=0)
data = data[columns]
# print(data.head())

# Tweet Data file processing ::::::::::::::::::::::::::::::::::::::::::::::::::::
def process_tweet_data(file):

    tweet_columns = ['text', 'retweet_count', 'favorite_count',	'follower_count',	'account']
    # dtypes = {'created_date': 'datetime', 'text': 'str', 'retweet_count': 'int','follower_count':'int','account':'str'}
    # parse_dates = ['created_date']

    # tweet_data = pd.read_csv(TRAIN_TWEETER, parse_dates=parse_dates, encoding="iso-8859-1")
    tweet_data = pd.read_csv(file, index_col=1, encoding="iso-8859-1")
    tweet_data = tweet_data[tweet_columns]
    # print(tweet_data.head())
    # tweet_data.rename(index = {'created_date': 'date'}, inplace = True)
    tweet_data.index.names = ['date']

    def sentiment_calc(text):
        try:
            return TextBlob(text).sentiment
        except:
            return None

    tweet_data['sentiment'] = tweet_data['text'].apply(sentiment_calc)

    def sentiment_split1(x):
        try:
            return x[0]
        except:
            return None

    def sentiment_split2(x):
        try:
            return x[1]
        except:
            return None  
    
    tweet_data['pol'] = tweet_data['sentiment'].apply(sentiment_split1)
    tweet_data['sub'] = tweet_data['sentiment'].apply(sentiment_split2)

    # tweet_data['date'] = tweet_data.index
    # tweet_data = tweet_data.drop(tweet_data[tweet_data.created_date == '5'].index)

    # def convert_date(x):
    #     try:
    #         return x.date()
    #     except (ValueError, TypeError):
    #         return None

    # print(tweet_data.head())
    # tweet_data['date'] = tweet_data['created_date'].apply(convert_date)

    clean_tw_data = tweet_data[['retweet_count','favorite_count', 'follower_count', 'account', 'pol', 'sub']]
    # clean_tw_data.set_index('date')

    clean_tw_data.drop(['account'], axis=1, inplace=True)
    clean_tw_data.drop(['favorite_count'], axis=1, inplace=True)

    return clean_tw_data
# -----------------------------------------------------------------------------------------

train_tweet_data = process_tweet_data(TRAIN_TWEETER)
test_tweet_data  = process_tweet_data(TEST_TWEETER)

print(train_tweet_data.head())

# Merge Tweet Data with Timeseries Data
data_close = data[columns]
data_w_tweets = train_tweet_data.merge(data_close, left_index=True, right_index=True)
data_w_tweets.head()