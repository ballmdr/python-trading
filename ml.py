#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'FXCM/Python Trading'))
	print(os.getcwd())
except:
	pass

#%%

import fxcmpy
import time
from datetime import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler, scale, StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import pandas as pd
import numpy as np
import talib as ta
import os

os_path = os.getcwd()


def getNewPrice():
    global price
    # update pricedata on first attempt
    new_price = con.get_candles(symbol, period=timeframe, number=n_price)

    if new_price.index.values[-1] != price.index.values[-1]:
        price = new_price
        return True

    counter = 0
    # If data is not available on first attempt, try up to 3 times to update pricedata
    while new_price.index.values[-1] == price.index.values[-1] and counter < 5:
        print("No updated prices found, trying again in 10 seconds...")
        writeLog("No updated prices found, trying again in 10 seconds...")
        time.sleep(10)
        new_price = con.get_candles(symbol, period=timeframe, number=n_price)
        counter += 1
    if new_price.index.values[-1] != price.index.values[-1]:
        price = new_price
        return True
    else:
        return False

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True,feat_name=None):
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'{feat_name[j]}(t-{i})' for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'{feat_name[j]}(t)' for j in range(n_vars)]
        else:
            names += [f'{feat_name[j]}(t+{i})' for j in range(n_vars)]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    return agg

def predictSignal():

    df = pd.DataFrame()
    df['Open'] = (price.askopen + price.bidopen) / 2
    df['High'] = (price.askhigh + price.bidhigh) / 2
    df['Low'] = (price.asklow + price.bidlow) / 2
    df['Close'] = (price.askclose + price.bidclose) / 2
    df['Volume'] = price.tickqty
    df.index = price.index

    df['Returns'] = df.Close.pct_change()
    df['Linear_regression'] = ta.LINEARREG(df.Close, timeperiod=14)
    df['Linear_angle'] = ta.LINEARREG_ANGLE(df.Close, timeperiod=14)
    df['Linear_slope'] = ta.LINEARREG_SLOPE(df.Close, timeperiod=14)
    df['Linear_intercept'] = ta.LINEARREG_INTERCEPT(df.Close, timeperiod=14)

    df['body_candle'] = df.Open - df.Close
    df['high_low'] = df.High - df.Low
    macd, macdsignal, macdhist = ta.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macdsignal'] = macdsignal
    df['macdhist'] = macdhist
    df['ma35'] = ta.SMA(df['Close'].values, timeperiod=35)
    df['range_ma35'] = df.Close - df.ma35
    df['ma200'] = ta.SMA(df['Close'].values, timeperiod=200)
    df['Returns'] = np.log(df.Close/df.Close.shift(1))
    df['ATR'] = ta.ATR(df['High'].values, df['Low'], df['Close'], timeperiod=14)
    df['ADX'] = ta.ADX(df.High, df.Low, df.Close, timeperiod=14)
    df['CCI'] = ta.CCI(df.High, df.Low, df.Close, timeperiod=14)
    df['MOM'] = ta.MOM(df.Close, timeperiod=10)
    df['RSI'] = ta.RSI(df.Close, timeperiod=14)

    df['Median'] = ta.MEDPRICE(df.High, df.Low)
    df['STD'] = np.std(df.Close)
    df['Pearson_coef'] = ta.CORREL(df.High, df.Low, timeperiod=30)
    df['Beta'] = ta.BETA(df.High, df.Low, timeperiod=5)
    df['obv'] = ta.OBV(df.Close, df.Volume)
    df['trendmode'] = ta.HT_TRENDMODE(df.Close)
    df['sine'], df['leadsine'] = ta.HT_SINE(df.Close)
    df['avgprice'] = ta.AVGPRICE(df.Open, df.High, df.Low, df.Close)
    df['typical_price'] = ta.TYPPRICE(df.High, df.Low, df.Close)
    df['weight_close'] = ta.WCLPRICE(df.High, df.Low, df.Close)
    df['aroondown'], df['aroonup'] = ta.AROON(df.High, df.Low, timeperiod=14)
    df['trendline'] = ta.HT_TRENDLINE(df.Close)
    df['kama'] = ta.KAMA(df.Close, timeperiod=35)
    df['midpoint'] = ta.MIDPRICE(df.High, df.Low, timeperiod=14)
    df['sar'] = ta.SAR(df.High, df.Low, acceleration=0, maximum=0)
    df['wma'] = ta.WMA(df.Close, timeperiod=35)
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
    df.dropna(inplace=True)

    reframed = series_to_supervised(df.values, sequence_len, 1, feat_name=df.columns)
    reframed = reframed.dropna()
    x = reframed.values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    predict_arr = pd.DataFrame(x[-1:])

    #predict_arr = scale(predict_arr.values).reshape(1,-1)
    #predict_arr = pca.transform(predict_arr)
    pred = model.predict(predict_arr)
    #print(predict_arr)
    print('predicted: ', pred)
    writeLog('predicted: ' + str(pred))
    #return pred
    if pred == 1:
        return True
    else:
        return False

def openPos(isBuy):
    global last_direction
    try:
        print('Open ', isBuy)
        opentrade = con.open_trade(symbol=symbol, is_buy=isBuy, amount=amount, time_in_force='GTC',order_type='AtMarket',is_in_pips=True,limit=limit, stop=stop)
    except:
        print('Open position %s not success' % symbol)
    else:
        print(opentrade)
        last_direction = isBuy

def closeAllPos():
    for i in range(len(con.open_pos)):
        #print('i = %i' % i)
        trade_id = con.get_open_trade_ids()[i]
        pos = con.get_open_position(trade_id)
        pos_amount = pos.get_amount()
        pos_symbol = pos.get_currency()
        if symbol == pos_symbol:
            con.close_trade(trade_id=trade_id, amount=pos_amount)
            print('close %s' % symbol)
            

def Update():
    global con, last_direction, symbol, amount
    if not con.is_connected():
        con = fxcmpy.fxcmpy(config_file = os_path + '/fxcm.cfg')
    if getNewPrice():
        print(str(datetime.now()) + " Got new prices -> Predicted Signal...")
        writeLog(str(datetime.now()) + " Got new prices -> Predicted Signal...")
        isBuy = predictSignal()
        #open_new = checkPosition(isBuy)
        #if open_new:
        poses = con.get_open_positions(kind='dataframe')
        print('len poses: ', len(poses))
        print('isBuy: ', isBuy)
        print('last_direction: ', last_direction)
        if (len(poses) == 0):
            print('Open first position')
            openPos(isBuy)
        elif (isBuy and not last_direction) or (not isBuy and last_direction):
            print('Close All')
            writeLog('Close All')
            closeAllPos()
            print(close_trade)
            openPos(isBuy)
        else:
            writeLog('Position exists')
            print('Position exists')
                    


#%%
def init():
    global con, price, last_direction, model
    print('init...')
    
    with open('log_final_data5_m30.pickle', 'rb') as file:
        model = pickle.load(file)
    
    con = fxcmpy.fxcmpy(config_file = os_path + '/fxcm.cfg')
    price = con.get_candles(symbol, period=timeframe, number=n_price)
    poses = con.get_open_positions(kind='dataframe')
    
    isBuy = predictSignal()
    if len(poses) > 0:
        print('have position: ', len(poses))
        last_direction = poses.iloc[-1].isBuy
        if last_direction != isBuy:
            print('Change Last direction: ', last_direction, isBuy)
            closeAllPos()
            openPos(isBuy)
        else:
            print('Not change direction')
    else:
        print('No position, Open new: ', isBuy)
        openPos(isBuy)
        
    print('Latest Num positions: ', len(con.open_pos))
    print('Latest Last direction: ', last_direction)
    
def writeLog(msg):
    file = open(mylog_path, 'a')
    file.write('\n')
    file.write(msg)
    file.close()
    
def main():
    while True:
        currenttime = datetime.now()
        #if True:
        if currenttime.second == 0 and currenttime.minute % 5 == 0:
            print('Time: ', currenttime.minute)
            writeLog('Time: ' + str(currenttime.minute))
        if currenttime.second == 0 and currenttime.minute % 30 == 0:
        #if True:
            writeLog('Time: ' + str(currenttime.minute))
            print('awakening...')
            writeLog('awakening...')
            Update()
            print('sleeping...')
            writeLog('sleeping...')
            time.sleep(1740)
        time.sleep(1)


#%%
symbol = 'AUD/NZD'
timeframe = "m30"	        # (m1,m5,m15,m30,H1,H2,H3,H4,H6,H8,D1,W1,M1)
amount = 1
account_id = '01041561'

price = None
n_price = 300
con = None
maxdd = 0
last_direction = None
max_amountK = 0
mylog_path = 'mylog.txt'

limit = None
stop = None

sequence_len = 0

#with open('pca_data3_m5.pickle', 'rb') as file:
#	pca = pickle.load(file)

#model = tf.keras.models.load_model('deep_25042019_data3_m5.h5')

init()

main()