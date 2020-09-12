import pandas as pd
import numpy as np
import sqlite3
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from matplotlib import style
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)



pd.read_csv("C:/Users/2kgod/OneDrive/바탕 화면/zigzag/._goods_info.csv")

connect = sqlite3.connect('c:/data/zigzag/zigzag_DB.db')
connect
query = "SELECT * FROM sqlite_master"

schema = pd.read_sql(query, connect)

for i in schema['sql']:
    print(i)
    
query = "SELECT * FROM 'order'"
order = pd.read_sql(query, connect)
order.head()
order.to_csv("c:/data/zigzag/order.csv",encoding='utf-8',index=False)

query = "SELECT * FROM 'good'"
good = pd.read_sql(query, connect)
good.head()
good.to_csv("c:/data/zigzag/good.csv",encoding='utf-8',index=False)

query = "SELECT * FROM 'shop'"
shop = pd.read_sql(query, connect)
shop.head()
shop.to_csv("c:/data/zigzag/shop.csv",encoding='utf-8',index=False)

query = "SELECT * FROM 'log'"
log = pd.read_sql(query, connect)
log.head()
log.to_csv("c:/data/zigzag/log.csv",encoding='utf-8',index=False)


query = "SELECT * FROM 'user'"
user = pd.read_sql(query, connect)
user.head()
user.to_csv("c:/data/zigzag/user.csv",encoding='utf-8',index=False)


#쇼핑몰 별 매출 TOP 10
s = order['price'].groupby(order['shop_id']).sum().reset_index()
s.columns=['shop_id','price_sum']
c = order['price'].groupby(order['shop_id']).count().reset_index()
c.columns=['shop_id','price_count']
top_10 = pd.merge(s,c,on='shop_id').sort_values('price_sum',ascending = False).head(10)


plt.figure(figsize=[10,7])
sns.barplot(data=top_10,x='shop_id',y='price_sum',order=top_10['shop_id'])



#판매액 시계열
order['timestamp'] = pd.to_datetime(order['timestamp'])
order.dtypes
plt.figure(figsize=[15,4])
sns.lineplot(data=order,x='timestamp',y='price')

#판매액 시간별 groupby
hour_price = order['price'].groupby(order['timestamp'].dt.hour).sum().reset_index()
plt.figure(figsize=[15,4])
sns.pointplot(data=hour_price,x='timestamp',y='price')

#연령대,시간대 별 매출
order_user = pd.merge(order,user,on='user_id')

lst = []
for i in order_user['age']:
    if (i>10)&(i<20):
        lst.append('10대')
    elif (i>20)&(i<30):
        lst.append('20대')
    elif (i>30)&(i<40):
        lst.append('30대')
    elif (i>40)&(i<50):
        lst.append('40대')
    else:
        lst.append('Unknown')
order_user['ages']=lst
t_10 = order_user[order_user['ages']=='10대']['price'].groupby(order['timestamp'].dt.hour).sum().reset_index()
t_20 = order_user[order_user['ages']=='20대']['price'].groupby(order['timestamp'].dt.hour).sum().reset_index()
t_30 = order_user[order_user['ages']=='30대']['price'].groupby(order['timestamp'].dt.hour).sum().reset_index()

t_10.to_csv("c:/data/zigzag/t_10.csv",encoding='utf-8',index=False)
t_20.to_csv("c:/data/zigzag/t_20.csv",encoding='utf-8',index=False)
t_30.to_csv("c:/data/zigzag/t_30.csv",encoding='utf-8',index=False)

t_10 = pd.read_csv("c:/data/zigzag/t_10.csv")
t_20 = pd.read_csv("c:/data/zigzag/t_20.csv")
t_30 = pd.read_csv("c:/data/zigzag/t_30.csv")

t_10['price'].plot(figsize=(15,4),marker="o",color='red',linewidth=2,label='10대')
t_20['price'].plot(figsize=(15,4),marker="o",color='green',linewidth=2,label='20대')
t_30['price'].plot(figsize=(15,4),marker="o",color='blue',linewidth=2,label='30대')
plt.xticks(range(0,24,1),size=8)
plt.legend()


#쇼핑몰 별 구매 나이대 boxplot
oup = pd.merge(order_user,top_10,on='shop_id')
oup.drop(oup[oup['age']==-1].index,inplace=True)
plt.figure(figsize=[10,6])
sns.boxplot(x='shop_id',y='age',data=oup)

#구매전환률 구하기
log['timestamp'] = pd.to_datetime(log['timestamp'])
order_copy = order.copy()
order_copy['event_origin'] = order_copy['shop_id']
order_copy['event_name'] = 'purchase'
order_copy['event_goods_id'] = order_copy['goods_id']
order_copy = order_copy[['timestamp', 'user_id','event_origin','event_name', 'event_goods_id', 'price']]
order_copy.head()


log['event_name'].value_counts()


log['event_name'].unique()
log_order = pd.concat([log,order_copy])
log_order=log_order.sort_values('timestamp').reset_index(drop=True)

goods_funnel = pd.pivot_table(data = log_order, 
                              index =  'event_goods_id', columns ='event_name' ,
                              values = 'timestamp', aggfunc = 'count').sort_values(
                              by = ['enter_browser', 'purchase'], 
                              ascending=False)

goods_funnel = goods_funnel[['enter_browser', 'add_my_goods','remove_my_goods', 'purchase']] 

good_shop = pd.merge(good,shop,on='shop_id')


good_shop_funnel = pd.merge(good_shop,goods_funnel,left_on ='goods_id',right_on='event_goods_id')

goods = good_shop_funnel[['name', 'goods_id', 'shop_id', 'style','enter_browser', 'add_my_goods', 'remove_my_goods', 'purchase']]
goods = goods.groupby('name')[['enter_browser','purchase']].sum().sort_values(['enter_browser','purchase'],ascending=False)
goods['구매전환율(%)'] = round(goods['purchase']/goods['enter_browser']*100,2)
goods['이탈률(%)'] = 100-goods['구매전환율(%)']

g_top20 = goods.head(20)
goods.index
x = np.arange(len(g_top20.index))
plt.figure(figsize=(15,7))
p1 = plt.bar(x, g_top20['enter_browser'],width=0.4)
p2 = plt.bar(x, g_top20['purchase'],width=0.4)
plt.title('쇼핑몰 별 유입수/구매수 TOP 20',fontsize=14)
plt.xlabel('shop_name', fontsize=10)
plt.xticks(x + 0.35 / 2, g_top20.index.unique(), rotation = 90)
plt.legend((p1[0], p2[0]), ('유입수','구매수'), fontsize=10)
plt.show()

g_top20 = goods.head(20).sort_values(['구매전환율(%)','이탈률(%)'],ascending=False)
x = np.arange(len(g_top20.index))
plt.figure(figsize=(15,7))
p3 = plt.bar(x, g_top20['구매전환율(%)'],  width=0.4,color='g')
plt.title('구매전환율(%)',fontsize=14)
plt.xlabel('shop_name', fontsize=10)
plt.xticks(x + 0.35 / 2, g_top20.index.unique(), rotation = 90)
plt.show()

#구매전환률이 가장 낮은 것고 높은 것의 카테고리가 다르므로 카테고리에 따른 구매전환률의 차이가 있을까 ?
shop[(shop['name']=='Angela')|(shop['name']=='Joan')]
 

good_category
good_category.index
x = np.arange(len(good_category.index))
plt.figure(figsize=(15,7))
p1 = plt.bar(x, good_category['enter_browser'],width=0.4)
p2 = plt.bar(x, good_category['purchase'],width=0.4)
plt.title('카테고리별 유입수/구매수',fontsize=14)
plt.xlabel('카테고리', fontsize=10)
plt.xticks(x + 0.35 / 2, good_category.index.unique(), rotation = 90)
plt.legend((p1[0], p2[0]), ('유입수','구매수'), fontsize=10)
plt.show()


good_category_sort=good_category.sort_values('구매전환율(%)',ascending=False)
x = np.arange(len(good_category_sort.index))
plt.figure(figsize=(15,7))
p3 = plt.bar(x, good_category_sort['구매전환율(%)'],  width=0.4,color='g')
plt.title('구매전환율(%)',fontsize=14)
plt.xlabel('shop_name', fontsize=10)
plt.xticks(x + 0.35 / 2, good_category_sort.index.unique(), rotation = 90)
plt.show()

#검색 
log_user = pd.merge(log,user,on='user_id')
log_user = log_user[~log_user['event_origin'].isin(['shops_ranking', 'shops_bookmark', 'my_goods'])]
log_user['keyword'] = log_user['event_origin'].apply(lambda x: x.split('/')[1] if type(x) != int else None)
log_user['event_origin'] = log_user['event_origin'].apply(lambda x: x.split('/')[0] if type(x) != int else None)

def make_g(i):
    if (i>10)&(i<20):
        return '10대'
    elif (i>20)&(i<30):
        return '20대'
    elif (i>30)&(i<40):
        return '30대'
    elif (i>40)&(i<50):
        return '40대'
    else:
        lst.append('Unknown')
        
log_user['연령대'] = log_user['age'].map(make_g)
plt.figure(figsize=(7,7))
log_user['연령대'].value_counts().plot(kind='bar',width=0.3)

def goods_s(age):
    return log_user[(log_user['event_origin']=='goods_search_result')&(log_user['연령대']==age)]['keyword'].value_counts()

def category(age):
    return log_user[(log_user['event_origin']=='category_search_result')&(log_user['연령대']==age)]['keyword'].value_counts()

ages=['10대','20대','30대','40대']
for i in ages:
    goods_s(i).to_csv("c:/data/zigzag/goods_s_{}.csv".format(i))
    
for i in ages:
    category(i).to_csv("c:/data/zigzag/category_{}.csv".format(i))
    
    
#연관분석
    
df = pd.merge(order,shop,on = 'shop_id')
df
style_list = ['페미닌', '모던시크', '심플베이직', '러블리', '유니크', '미시스타일', '캠퍼스룩', '빈티지', '섹시글램', '스쿨룩', '로맨틱', '오피스룩',
              '럭셔리', '헐리웃스타일', '심플시크', '키치', '펑키', '큐티', '볼드&에스닉' ]
for style in style_list:
    shop[f"{style}"] = shop['style'].str.contains(style)
    
merged = (
    order.merge(shop, on='shop_id')
             .merge(user, on='user_id')
)
print(merged.shape)
merged.head(3)

recommend_df = merged.iloc[:,9:-2]
recommend_df=recommend_df.fillna(False)


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

f = apriori(recommend_df, min_support=0.05, use_colnames=True)
f.sort_values('support',ascending = False)
association_rules(f,metric = 'confidence',min_threshold=0.3)

