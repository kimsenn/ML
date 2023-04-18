import pandas as pd

df = pd.read_csv("chipotle.tsv",delimiter='\t')
print(df.columns)
#print(str(df))
# print(df.info())

# item_name을 order_id에 따라 그룹짓기 
# order_id가 1834명, 총 상품 50개
df_tmp=df[['order_id','item_name']]
df_tmp_arr=[[]for i in range(1835)]
num=0
for i in df_tmp['item_name'] :
    df_tmp_arr[df_tmp['order_id'][num]].append(i)
    num+=1
df_tmp['item_name']

df_tmp_arr.pop(0)
num=0
for i in df_tmp_arr :
    df_tmp_arr[num] = list(set(df_tmp_arr[num]))
    num+=1
# df_tmp_arr

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

te = TransactionEncoder()
te_ary = te.fit(df_tmp_arr).transform(df_tmp_arr)

# 모든 데이터에 대해, 각 리스트(행마다) 존재하면 True, 없으면 False로 나타냄
df = pd.DataFrame(te_ary, columns=te.columns_)

# 지지도 출력
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
frequent_itemsets

# 향상도가 최소 1 이상인 연관규칙,  antecedents : 조건절 / consequents : 결과절
association_rules(frequent_itemsets, metric="lift", min_threshold=1)
