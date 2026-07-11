
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

print("Bank Fraud Detection")

np.random.seed(1)

# making fake data since i dont have real bank data
normal_amount = np.random.randint(100, 5000, 900)
normal_time = np.random.randint(8, 22, 900)
normal_risk = np.random.uniform(0, 0.3, 900)
normal_label = [0] * 900

fraud_amount = np.random.randint(3000, 20000, 100)
fraud_time = np.random.randint(0, 5, 100)
fraud_risk = np.random.uniform(0.7, 1, 100)
fraud_label = [1] * 100

amount = list(normal_amount) + list(fraud_amount)
time = list(normal_time) + list(fraud_time)
risk = list(normal_risk) + list(fraud_risk)
label = normal_label + fraud_label

df = pd.DataFrame({
    'amount': amount,
    'time': time,
    'risk': risk,
    'label': label
})

df = df.sample(frac=1).reset_index(drop=True)

print("total rows", len(df))
print("fraud rows", df['label'].sum())

# fraud is very less compared to normal, dataset is imbalanced

plt.hist(df[df['label']==0]['amount'], bins=20, alpha=0.5, label='normal')
plt.hist(df[df['label']==1]['amount'], bins=20, alpha=0.5, label='fraud')
plt.legend()
plt.title('amount vs fraud')
plt.savefig('graph.png')
plt.close()

x = df[['amount', 'time', 'risk']]
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)

print("accuracy", acc)

cm = confusion_matrix(y_test, pred)
print(cm)
