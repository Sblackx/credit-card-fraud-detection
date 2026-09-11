import pandas as pd
import matplotlib.pyplot as plt


file = open('creditcard.csv', 'r')
df = pd.read_csv(file)

# print(df.shape)
# print(df.head())
# print(df['Class'].value_counts())

print(df.groupby('Class')['Amount'].describe())
print(df.groupby('Class')['Time'].describe())



# plt.figure(figsize=(8, 5))
# df.boxplot(column='Amount', by='Class')
# plt.title('Transaction Amount by Class (0=Normal, 1=Fraud)')
# plt.suptitle('')  # removes the default ugly auto-title pandas adds
# plt.xlabel('Class')
# plt.ylabel('Amount')
# plt.yscale('log')
# plt.show()


df['Hour'] = (df['Time'] // 3600) % 24

fraud_by_hour = df.groupby('Hour')['Class'].mean()

#from visuilizing and reading the data, we can see that most of
#fraud trans happens between 2 to 4 (am)
plt.figure(figsize=(10, 5))
fraud_by_hour.plot(kind='bar')
plt.title('Fraud Rate by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Fraud Rate')
plt.show()



#compare each one of the classes visualization
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

fraud['Hour'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='red')
axes[0].set_title('Fraud Transactions by Hour')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Count')

normal['Hour'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='blue')
axes[1].set_title('Normal Transactions by Hour')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()











