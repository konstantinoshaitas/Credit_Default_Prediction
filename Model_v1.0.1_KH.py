#PACKAGE IMPORT
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sc
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, mean_squared_error, mean_absolute_error, explained_variance_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree



#STYLE FOR PLOTTING GRAPHS

sns.set_style("darkgrid")
sns.set_context("paper", font_scale = 0.75)



#IMPORT DATA

df_og = pd.read_csv(r"C:\Users\koko\Desktop\ML\Credit_24\data\train.csv")
df_og = df_og.drop(columns="Unnamed: 0")
df_og.columns = df_og.columns.str.lower() #Main DataFrame



#DATA PREPROCESSING

df = df_og.dropna()         #Drop Null Values - Main Dataframe to work from
df = df.drop_duplicates()   #Drop Duplicate Rows
df = df[df['age'] >= 18]    #Clean Age - Only keep over 18 year olds.

print(df.info())
# print(df.duplicated().value_counts())
# print(f'number of people over 95:', np.sum(df['age'] > 95))



#SPLIT
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)   #NOTE: y_test is equivalent to y_true in this case.



#NORMALISATION

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#LOGISTIC MODEL

model_l = LogisticRegression()                          # model selection
model_l.fit(X_train, y_train)                           # train logistic regression on train
y_pred = model_l.predict(X_test)                        # predict on test data
y_pred_decf = model_l.decision_function(X_test)         # decf stands for decision function




#TESTING - Accuracy, ROC_Curve and ROC_AUC test results

print(f'accuracy score:', accuracy_score(y_test, y_pred)) # accuracy score = % of correct predictions


logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_decf)       # fpr = false positive rate, tpr = true positive rate
auc_logistic = auc(logistic_fpr, logistic_tpr)
logistic_auc_score = roc_auc_score(y_test, y_pred_decf)
print(f'logistic_auc_score: {logistic_auc_score}')


plt.figure(figsize=(4,3), dpi=128)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(logistic_fpr, logistic_tpr, label='Logistic (auc = %0.3f)' % auc_logistic)

plt.legend()
plt.show()