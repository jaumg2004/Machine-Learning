import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

with open(r'C:\Users\Jaum\Desktop\download\download\3.Algoritmos de Machine Learning\Transacoes.txt') as f:
    transactions = [line.strip().split(',') for line in f.readlines()]

print(transactions)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
print(rules)