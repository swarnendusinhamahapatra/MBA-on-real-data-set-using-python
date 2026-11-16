# assignment_1_mba.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
pd.set_option("display.max_columns", None)

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
mba = pd.read_csv(DATA_DIR / "grocery_transactions.csv")
print("Dataset head:")
print(mba.head())

# Convert rows into transaction lists
transactions = mba.groupby("TransactionID")["Item"].apply(list).tolist()

te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(encoded, columns=te.columns_)

print("Number of transactions:", len(transactions))
print("Number of unique items:", len(te.columns_))
print(basket_df.head())

# Find frequent itemsets
frequent_itemsets = apriori(
    basket_df,
    min_support=0.10,
    use_colnames=True
).sort_values("support", ascending=False)

print("\nTop Frequent Itemsets:")
print(frequent_itemsets.head(10))

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.30
)

rules = rules.sort_values(
    ["lift", "confidence"],
    ascending=False
)

print("\nTop Association Rules:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# Visualize the most frequent individual items
item_frequency = basket_df.sum().sort_values(ascending=False).head(10)

item_frequency.plot(kind="bar", figsize=(10, 5), title="Top 10 Items by Transaction Frequency")
plt.xlabel("Item")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()