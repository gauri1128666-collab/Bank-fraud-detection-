
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("Starting Bank Fraud Detection Project...")
print("------------------------------------------")


np.random.seed(10)
total = 5000

# normal transactions
normal_data = {
    'amount': np.random.randint(100, 5000, 4900),
    'time_of_day': np.random.randint(8, 22, 4900),      # normal hours
    'risk_score': np.random.uniform(0.0, 0.3, 4900),
    'is_foreign': np.random.choice([0, 1], 4900, p=[0.9, 0.1]),
    'label': [0] * 4900   # 0 = not fraud
}


fraud_data = {
    'amount': np.random.randint(3000, 20000, 100),
    'time_of_day': np.random.randint(0, 5, 100),         # late night = fraud
    'risk_score': np.random.uniform(0.7, 1.0, 100),
    'is_foreign': np.random.choice([0, 1], 100, p=[0.2, 0.8]),
    'label': [1] * 100    # 1 = fraud
}

df1 = pd.DataFrame(normal_data)
df2 = pd.DataFrame(fraud_data)
df  = pd.concat([df1, df2], ignore_index=True)


df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total transactions : {len(df)}")
print(f"Fraud cases        : {df['label'].sum()}")
print(f"Normal cases       : {(df['label']==0).sum()}")
print()




plt.figure(figsize=(12, 4))

# graph 1 - how many fraud vs normal
plt.subplot(1, 3, 1)
counts = df['label'].value_counts()
plt.bar(['Normal', 'Fraud'], counts.values, color=['blue', 'red'])
plt.title('Normal vs Fraud')
plt.ylabel('Count')


plt.subplot(1, 3, 2)
plt.hist(df[df['label']==0]['amount'], bins=30, alpha=0.6, color='blue', label='Normal')
plt.hist(df[df['label']==1]['amount'], bins=30, alpha=0.6, color='red', label='Fraud')
plt.title('Transaction Amount')
plt.xlabel('Amount')
plt.legend()


plt.subplot(1, 3, 3)
plt.hist(df[df['label']==0]['time_of_day'], bins=20, alpha=0.6, color='blue', label='Normal')
plt.hist(df[df['label']==1]['time_of_day'], bins=20, alpha=0.6, color='red', label='Fraud')
plt.title('Time of Transaction')
plt.xlabel('Hour')
plt.legend()

plt.tight_layout()
plt.savefig('fraud_graphs.png')
plt.close()
print("Graphs saved as fraud_graphs.png")



features = ['amount', 'time_of_day', 'risk_score', 'is_foreign']
X = df[features]
y = df['label']

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data size : {len(X_train)}")
print(f"Testing data size  : {len(X_test)}")
print()




# Model 1 - Logistic Regression (simple model i learned in class)
print("Training Logistic Regression...")
model1 = LogisticRegression()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
acc1  = accuracy_score(y_test, pred1)
print(f"Logistic Regression Accuracy : {acc1*100:.2f}%")

# Model 2 - Random Forest (better model)
print("Training Random Forest...")
model2 = RandomForestClassifier(n_estimators=50, random_state=42)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)
acc2  = accuracy_score(y_test, pred2)
print(f"Random Forest Accuracy       : {acc2*100:.2f}%")
print()


print("Using Random Forest as final model")
print()
print("Classification Report:")
print(classification_report(y_test, pred2, target_names=['Normal', 'Fraud']))




cm = confusion_matrix(y_test, pred2)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.colorbar()
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.yticks([0, 1], ['Normal', 'Fraud'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha='center', va='center', color='black', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix saved as confusion_matrix.png")
print()


# 
print("------------------------------------------")
print("Testing on new transactions:")
print("------------------------------------------")


t1 = pd.DataFrame([[500, 14, 0.1, 0]], columns=features)
r1 = model2.predict(t1)[0]
p1 = model2.predict_proba(t1)[0][1]
print(f"Transaction 1 (amount=500, time=2pm, risk=0.1)")
print(f"  Result : {'FRAUD' if r1==1 else 'NORMAL'} | Fraud chance: {p1*100:.1f}%")


t2 = pd.DataFrame([[15000, 2, 0.95, 1]], columns=features)
r2 = model2.predict(t2)[0]
p2 = model2.predict_proba(t2)[0][1]
print(f"Transaction 2 (amount=15000, time=2am, risk=0.95, foreign)")
print(f"  Result : {'FRAUD' if r2==1 else 'NORMAL'} | Fraud chance: {p2*100:.1f}%")

print()
print("------------------------------------------")
print("Project Complete!")
print(f"Best Model    : Random Forest")
print(f"Best Accuracy : {acc2*100:.2f}%")
print("------------------------------------------")
