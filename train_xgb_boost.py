import os
import pandas as pd
from gan import GAN
import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import xgboost as xgb
import joblib
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"]=""

class TrainXGBBoost:

    def __init__(self, num_historical_days, days=10, pct_change=0):  #过去num_historical_days天的数据预测未来第days的涨跌
        self.data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
        assert os.path.exists('./models/checkpoint')
        gan = GAN(num_features=5, num_historical_days=num_historical_days,
                        generator_input_size=200, is_train=False)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            with open('./models/checkpoint', 'rb') as f:
                model_name = str(next(f)).split('"')[1]
            saver.restore(sess, "./models/{}".format(model_name))
            files = [os.path.join('./stock_data', f) for f in os.listdir('./stock_data')]
            for file in files:
                print(file)
                ext = os.path.splitext(file)[-1]
                if ext != '.csv':
                    continue
                #Read in file -- note that parse_dates will be need later
                df = pd.read_csv(file, index_col='trade_date', parse_dates=True)
                df = df[['open','high','low','close','vol']]
                # #Create new index with missing days
                # idx = pd.date_range(df.index[-1], df.index[0])
                # #Reindex and fill the missing day with the value from the day before
                # df = df.reindex(idx, method='bfill').sort_index(ascending=False)
                #Normilize using a of size num_historical_days
                sss = 1/(df.close.pct_change(days)+1) - 1   #以当前为基准，days天之后的真实涨幅，最近的days天是没有的
                lll = lambda x: 0 if pd.isnull(x) else int(x > pct_change/100.0)  #写的严谨点，最近days天不知道days天之后的涨幅，算作0
                labels = sss.map(lll)  #正的越多说明涨的越多
                df = ((df -
                df.rolling(num_historical_days).mean().shift(-num_historical_days+1))
                /(df.rolling(num_historical_days).max().shift(-num_historical_days+1)
                -df.rolling(num_historical_days).min().shift(-num_historical_days+1)))
                df['labels'] = labels    #labels记录days天之后的涨幅是否大于5个点，这就是回归的目标，如果预测的足够准确，则可以实战试试
                #Drop the last 10 day that we don't have data for
                df = df.dropna()
                #Hold out the last year of trading for testing
                #最近一年的数据进行测试
                test_df = df[:365]
                #Padding to keep labels from bleeding
                #400天前的数据进行训练
                df = df[400:]
                #This may not create good samples if num_historical_days is a
                #mutliple of 7
                data = df[['open', 'high', 'low', 'close', 'vol']].values
                labels = df['labels'].values
                for i in range(num_historical_days, len(df), num_historical_days):  #步长num_historical_days是不是太长了
                    starti = i-num_historical_days #窗口起始索引
                    endi   = i                     #窗口结束索引
                    wini   = data[starti:endi]     #窗口数据
                    labi   = labels[starti]        #窗口对应的标识，未来第days天的涨幅

                    features = sess.run(gan.features, feed_dict={gan.X:[wini]})  #对窗口num_historical_days天数据使用gan进行处理
                    self.data.append(features[0])
                    self.labels.append(labi)

                    print(features[0])
                    print(labi)
                data = test_df[['open', 'high', 'low', 'close', 'vol']].values
                labels = test_df['labels'].values
                for i in range(num_historical_days, len(test_df), 1):              #感觉步长1是ok的
                    starti = i-num_historical_days
                    endi   = i
                    wini   = data[starti:endi]
                    labi   = labels[starti]

                    features = sess.run(gan.features, feed_dict={gan.X:[wini]})
                    self.test_data.append(features[0])
                    self.test_labels.append(labi)



    def train(self):
        params = {}
        params['objective'] = 'multi:softprob'
        params['eta'] = 0.01
        params['num_class'] = 2
        params['max_depth'] = 20
        params['subsample'] = 0.05
        params['colsample_bytree'] = 0.05
        params['eval_metric'] = 'mlogloss'
        #params['scale_pos_weight'] = 10
        #params['silent'] = True
        #params['gpu_id'] = 0
        #params['max_bin'] = 16
        #params['tree_method'] = 'gpu_hist'

        train = xgb.DMatrix(self.data, self.labels)
        test = xgb.DMatrix(self.test_data, self.test_labels)

        watchlist = [(train, 'train'), (test, 'test')]
        clf = xgb.train(params, train, 10000, evals=watchlist, early_stopping_rounds=1000)
        joblib.dump(clf, 'models/clf.pkl')
        ppp = clf.predict(test, iteration_range=(0, clf.best_iteration + 1))
        lll = lambda x: int(x[1] > 0.5)
        mmm = list(map(lll, ppp))
        cm = confusion_matrix(self.test_labels, mmm)
        print(cm)
        plot_confusion_matrix(cm, ['Down', 'Up'], normalize=True, title="Confusion Matrix")
        #xgb.plot_importance(clf)


boost_model = TrainXGBBoost(num_historical_days=20, days=10, pct_change=0)  #国内股市波动比较小，pct_change不能设置太大，
boost_model.train()
