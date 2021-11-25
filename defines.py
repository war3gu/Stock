

stock_features_0 = ['open','high','low','close','vol','amount','pre_close','change','pct_chg']

stock_features_1 = ['open','high','low','close','vol']

def_stock_features = stock_features_0

def_num_features = len(def_stock_features)         #属性池长度

def_num_historical_days = 20                       #一组数据天数

def_dummy_steps = def_num_historical_days - 2      #卷积会导致丢失2个维度

def_predict_days = 10                              #预测多少天后的股价


stock_arr_yiyao = ['600196.sh','600276.sh','002821.sz','002001.sz']

stock_arr_yiyao1 = ['600196.sh','000963.sz','600276.sh','600079.sh','002422.sz','600380.sh','600420.sh','002001.sz',
                    '600664.sh','000513.sz','600267.sh','600673.sh','600062.sh','600216.sh','000739.sz','002793.sz']

stock_arr_jiadian = ['000333.sz', '600690.sh', '000651.sz']

stock_arr_0 = ['000333.sz', '600196.sh', '600519.sh', '600104.sh']

def_stock_arr = stock_arr_yiyao




