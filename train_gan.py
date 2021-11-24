import os
import pandas as pd
from gan import GAN
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

random.seed(42)
class TrainGan:

    def __init__(self, num_historical_days, batch_size=128):
        self.batch_size = batch_size
        self.data = []
        files = [os.path.join('./stock_data', f) for f in os.listdir('./stock_data')]
        for file in files:
            print(file)
            ext = os.path.splitext(file)[-1]
            if ext != '.csv':
                continue
            #Read in file -- note that parse_dates will be need later
            df = pd.read_csv(file, index_col='trade_date', parse_dates=True)
            print('record')
            df = df[['open','high','low','close','vol']]
            # #Create new index with missing days
            # idx = pd.date_range(df.index[-1], df.index[0])
            # #Reindex and fill the missing day with the value from the day before
            # df = df.reindex(idx, method='bfill').sort_index(ascending=False)
            #Normilize using a of size num_historical_days
            roll = df.rolling(num_historical_days)   #定义num_historical_days个元素的移动窗口

            rollMean = roll.mean()                   #前num_historical_days-1个皆为nan
            rollMeanShi = rollMean.shift(-num_historical_days+1) #将nan行移除掉。此处的num_historical_days应该要减一，等以后模型稳定了再试试

            rollMax = roll.max()                     #前num_historical_days-1个皆为nan
            rollMaxShi = rollMax.shift(-num_historical_days+1)

            rollMin = roll.min()
            rollMinShi = rollMin.shift(-num_historical_days+1)

            uu = (df - rollMeanShi)
            dd = (rollMaxShi - rollMinShi)
            df = uu/dd      #使用移动窗口归一化
            #Drop the last 10 day that we don't have data for
            df = df.dropna()
            #Hold out the last year of trading for testing
            #Padding to keep labels from bleeding
            #取400天前的数据进行训练。前面的数据靠近现在,留作测试
            df = df[400:]
            #This may not create good samples if num_historical_days is a
            #mutliple of 7
            for i in range(num_historical_days, len(df), num_historical_days):
                starti = i-num_historical_days
                endi = i
                wini = df.values[starti:endi]     #num_historical_days个一组
                self.data.append(wini)

        self.gan = GAN(num_features=5, num_historical_days=num_historical_days,
                        generator_input_size=200)

    def random_batch(self, batch_size=128):
        batch = []
        while True:
            batch.append(random.choice(self.data))
            if (len(batch) == batch_size):
                yield batch
                batch = []

    def train(self, print_steps=100, display_data=100, save_steps=500):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        G_loss = 0
        D_loss = 0
        G_l2_loss = 0
        D_l2_loss = 0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if os.path.exists('./models/checkpoint'):
            with open('./models/checkpoint', 'rb') as f:
                cc = str(next(f))
                dd = cc.split('"')
                model_name = dd[1]
                saver.restore(sess, "./models/{}".format(model_name))
        else:
            print('no checkpoint')
        for i, X in enumerate(self.random_batch(self.batch_size)):
            if i % 1 == 0:
                _, D_loss_curr, D_l2_loss_curr = sess.run([self.gan.D_solver, self.gan.D_loss, self.gan.D_l2_loss], feed_dict=
                        {self.gan.X:X, self.gan.Z:self.gan.sample_Z(self.batch_size, 200)})
                D_loss += D_loss_curr
                D_l2_loss += D_l2_loss_curr
            if i % 1 == 0:
                _, G_loss_curr, G_l2_loss_curr = sess.run([self.gan.G_solver, self.gan.G_loss, self.gan.G_l2_loss],
                        feed_dict={self.gan.Z:self.gan.sample_Z(self.batch_size, 200)})
                G_loss += G_loss_curr
                G_l2_loss += G_l2_loss_curr
            if (i+1) % print_steps == 0:
                print('Step={} D_loss={}, G_loss={}'.format(i, D_loss/print_steps - D_l2_loss/print_steps, G_loss/print_steps - G_l2_loss/print_steps))
                #print('D_l2_loss = {} G_l2_loss={}'.format(D_l2_loss/print_steps, G_l2_loss/print_steps))
                G_loss = 0
                D_loss = 0
                G_l2_loss = 0
                D_l2_loss = 0
            if (i+1) % save_steps == 0:
                saver.save(sess, './models/gan.ckpt', i)
            # if (i+1) % display_data == 0:
            #     print('Generated Data')
            #     print(sess.run(self.gan.gen_data, feed_dict={self.gan.Z:self.gan.sample_Z(1, 200)}))
            #     print('Real Data')
            #     print(X[0])


if __name__ == '__main__':
    gan = TrainGan(20, 128)
    gan.train()
