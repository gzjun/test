import pandas as pd
import matplotlib.pyplot as plt


path_train = '../input/train_411.csv'
path_test = '../input/test_411.csv'

train = pd.read_csv(path_train)
test = pd.read_csv(path_test)

train[['shop_id','item_id']].plot('b.')
plt.show()
test[['shop_id','item_id']].plot('go')
plt.show()

df_train = train.groupby(by=['user_id'], as_index=False)[['item_id']].count()
plt.plot(df_train['user_id'].values, df_train['is_trade'].values, 'g.')
df_test = test.groupby(by=['user_id'], as_index=False)[['item_id']].count()
plt.plot(df_test['user_id'].values, df_test['is_trade'].values, 'b.')
plt.show()


##从图中基本可以看出，训练集基本包括了测试集的内容
