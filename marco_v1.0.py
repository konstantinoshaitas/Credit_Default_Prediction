import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sc
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, mean_squared_error, mean_absolute_error, \
    explained_variance_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV

# LOADING IN DATA
data = pd.read_csv(r"C:\Users\koko\Desktop\ML\Credit_24\data\train.csv")
data.head(10)

data.info()

data = data.drop(columns="Unnamed: 0")
data.columns = data.columns.str.lower()
data.head(10)

data.describe()

dropped = data.dropna()
dropped.describe()

data[data.monthlyincome.isna()].describe()

##IMPUTING DATA TO REMEDY NAN VALUES IN monthlyincome VARIABLE
##
##
##
data_imputed = data.copy()
to_impute = 1040.018336 * 246.2368
data_imputed['monthlyincome'].fillna(to_impute, inplace=True)
data_imputed['numberofdependents'].fillna(0, inplace=True)
data_imputed.head(10)

data_imputed.describe()
data_imputed.info()

# SPLIT
y = data_imputed.iloc[:, 0]
X = data_imputed.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)  # NOTE: y_test is equivalent to y_true in this case.

# NORMALISATION

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# LOGISTIC MODEL

model_l = LogisticRegression()  # model selection
model_l.fit(X_train, y_train)  # train logistic regression on train
y_pred = model_l.predict(X_test)  # predict on test data
y_pred_decf = model_l.decision_function(X_test)  # decf stands for decision function

# TESTING - Accuracy, ROC_Curve and ROC_AUC test results

print(f'accuracy score:', accuracy_score(y_test, y_pred))  # accuracy score = % of correct predictions

logistic_fpr, logistic_tpr, threshold = roc_curve(y_test,
                                                  y_pred_decf)  # fpr = false positive rate, tpr = true positive rate
auc_logistic = auc(logistic_fpr, logistic_tpr)
logistic_auc_score = roc_auc_score(y_test, y_pred_decf)
print(f'logistic_auc_score: {logistic_auc_score}')

plt.figure(figsize=(4, 3), dpi=128)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(logistic_fpr, logistic_tpr, label='Logistic (auc = %0.3f)' % auc_logistic)

plt.legend()
plt.show()

imputer = IterativeImputer(max_iter=20, random_state=0)  # initializing IterativeImputer
imputer.fit(data)
imputed_data = imputer.transform(data)

data_iter_imp = pd.DataFrame(imputed_data, columns=data.columns)
data_iter_imp.head(10)

# SPLIT
y = data_iter_imp.iloc[:, 0]
X = data_iter_imp.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)  # NOTE: y_test is equivalent to y_true in this case.

# NORMALISATION

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# LOGISTIC MODEL

model_l = LogisticRegression()  # model selection
model_l.fit(X_train, y_train)  # train logistic regression on train
y_pred = model_l.predict(X_test)  # predict on test data
y_pred_decf = model_l.decision_function(X_test)  # decf stands for decision function

# TESTING - Accuracy, ROC_Curve and ROC_AUC test results

print(f'accuracy score:', accuracy_score(y_test, y_pred))  # accuracy score = % of correct predictions

logistic_fpr, logistic_tpr, threshold = roc_curve(y_test,
                                                  y_pred_decf)  # fpr = false positive rate, tpr = true positive rate
auc_logistic = auc(logistic_fpr, logistic_tpr)
logistic_auc_score = roc_auc_score(y_test, y_pred_decf)
print(f'logistic_auc_score: {logistic_auc_score}')

plt.figure(figsize=(4, 3), dpi=128)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(logistic_fpr, logistic_tpr, label='Logistic (auc = %0.3f)' % auc_logistic)

plt.legend()
plt.show()

# SPLIT

y = data_iter_imp.iloc[:, 0]
X = data_iter_imp.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)  # NOTE: y_test is equivalent to y_true in this case.

# DECISION TREE MODEL

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred = model_dt.predict(X_test)
y_pred_probs = model_dt.predict_proba(X_test)  # output is a 2 dimensional array,
# 1st column is the prob an observation belongs in the 0 class, 2nd is the prob it belongs to the positive class


# TESTING - Accuracy, ROC_Curve and ROC_AUC test results

print(f'accuracy score:', accuracy_score(y_test, y_pred))  # accuracy score = % of correct predictions

decision_fpr, decision_tpr, threshold = roc_curve(y_test, y_pred_probs[:,
                                                          1])  # selecting the probs of belonging to the positive class
auc_decision = auc(decision_fpr, decision_tpr)  # (roc and auc only function with the positive class probabilities)
decision_auc_score = roc_auc_score(y_test,
                                   y_pred_probs[:, 1])  # again, selecting the probs of belonging to the positive class
print(f'decision_auc_score: {decision_auc_score}')

plt.figure(figsize=(4, 3), dpi=128)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(decision_fpr, decision_tpr, label='Decision (auc = %0.3f)' % auc_decision)

plt.legend()
plt.show()

# SPLIT

y = data_iter_imp.iloc[:, 0]
X = data_iter_imp.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)  # NOTE: y_test is equivalent to y_true in this case.

# making a range of the size of the ensemble to compare scores
estimator_range = np.arange(60, 500, 20)

# arrays of models and scores for each training iteration
models = []
scores = []

for n_estimators in estimator_range:
    # Create bagging classifier
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)

    # Fit the model
    clf.fit(X_train, y_train)

    # Append the model and score to their respective list
    models.append(clf)
    scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))

# Generate the plot of scores against number of estimators
plt.figure(figsize=(9, 6))
plt.plot(estimator_range, scores)

# Adjust labels and font (to make visible)
plt.xlabel("n_estimators", fontsize=18)
plt.ylabel("score", fontsize=18)
plt.tick_params(labelsize=16)

# Visualize plot
plt.show()

# SPLIT

y = data_iter_imp.iloc[:, 0]
X = data_iter_imp.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)  # NOTE: y_test is equivalent to y_true in this case.

# BAGGING CLASSIFIER MODEL

model_bag = BaggingClassifier(n_estimators=400)
model_bag.fit(X_train, y_train)
y_pred = model_bag.predict(X_test)
y_pred_probs = model_bag.predict_proba(X_test)  # output is a 2 dimensional array,
# 1st column is the prob an observation belongs in the 0 class, 2nd is the prob it belongs to the positive class


# TESTING - Accuracy, ROC_Curve and ROC_AUC test results

print(f'accuracy score:', accuracy_score(y_test, y_pred))  # accuracy score = % of correct predictions

bagging_fpr, bagging_tpr, threshold = roc_curve(y_test, y_pred_probs[:,
                                                        1])  # selecting the probs of belonging to the positive class
auc_bagging = auc(bagging_fpr, bagging_tpr)  # (roc and auc only function with the positive class probabilities)
bagging_auc_score = roc_auc_score(y_test,
                                  y_pred_probs[:, 1])  # again, selecting the probs of belonging to the positive class
print(f'bagging_auc_score: {bagging_auc_score}')

plt.figure(figsize=(4, 3), dpi=128)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(bagging_fpr, bagging_tpr, label='Bagging (auc = %0.3f)' % auc_bagging)

plt.legend()
plt.show()
