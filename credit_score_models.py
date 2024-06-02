"""
0. PACKAGE
"""
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

"""
1. GRAPH AESTHETIC
"""

sns.set_style("white")
sns.set_context("talk", font_scale=.8)
palette = sns.color_palette()


"""
2. LOADING IN DATA
"""
data = pd.read_csv(r"C:\Users\koko\Desktop\ML\Credit_24\data\train.csv")
df_predictions = pd.read_csv(r'C:\Users\koko\Desktop\ML\Credit_24\data\test.csv')

"""
3. DATA PREPROCESSING
"""

data = data.drop(columns="Unnamed: 0")
data.columns = data.columns.str.lower()
df_drop = data.dropna()  # dropped null values - shown to be a worse solution than imputed missing values.

# IMPUTATION OF NULL VALUES (METHOD 1 - SEE REPORT FOR REFERENCE)
df_imputed = data.copy()
to_impute = 1040.018336 * 246.2368  # See attached report for impute methodology.
df_imputed['monthlyincome'].fillna(to_impute, inplace=True)  # Dealing with missing values.
df_imputed['numberofdependents'].fillna(0, inplace=True)  # Dealing with missing values.

# ITERATIVE IMPUTE (METHOD 2 - SEE REPORT FOR REFERENCES)
iter_imp = IterativeImputer(max_iter=20, random_state=0)  # initializing IterativeImputer
iter_imp.fit(data)
imputed_data = iter_imp.transform(data)
data_iter_imp = pd.DataFrame(imputed_data, columns=data.columns)

"""
4. LOGISTIC MODEL - IMPUTE (METHOD 1)
"""
# SPLIT METHOD 1 (TRAIN - TEST)
y = df_imputed.iloc[:, 0]  # Target Variable: y from first column
X = df_imputed.iloc[:, 1:]  # Input Features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# NOTE: y_test = y_true

# NORMALISATION
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# LOGISTIC MODEL 1
model_l = LogisticRegression()  # model selection
model_l.fit(X_train, y_train)  # train logistic regression on train
y_pred = model_l.predict(X_test)  # predict on test data
y_pred_decf = model_l.decision_function(X_test)  # decision function - necessary for roc_auc_score

# TESTING (ACCURACY, ROC_CURVE AND ROC_AUC_SCORE)
print(f'Accuracy Score Logistic 1:', accuracy_score(y_test, y_pred))  # accuracy score = % of correct predictions
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_decf)
# NOTE: fpr = false positive rate, tpr = true positive rate

auc_logistic = auc(logistic_fpr, logistic_tpr)
logistic_auc_score = roc_auc_score(y_test, y_pred_decf)
print(f'Logistic (Impute Method 1) AUC Score: {logistic_auc_score}')

plt.figure(figsize=(8, 6), dpi=100)
plt.title('ROC Curve Analysis', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold', labelpad=12)
plt.plot(logistic_fpr, logistic_tpr, label='Logistic 1 (AUC = %0.3f)' % auc_logistic, color=palette[0])
plt.legend()
plt.show()

"""
5. LOGISTIC MODEL - ITER IMPUTE (METHOD 2)
"""

# SPLIT METHOD 2 (TRAIN - TEST)

y = data_iter_imp.iloc[:, 0]
X = data_iter_imp.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# NORMALISATION

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# LOGISTIC MODEL 2

model_l2 = LogisticRegression()  # model selection
model_l2.fit(X_train, y_train)  # train logistic regression on train
y_pred = model_l2.predict(X_test)  # predict on test data
y_pred_decf = model_l2.decision_function(X_test)  # decision function - necessary for roc_auc_score

# TESTING (ACCURACY, ROC_CURVE AND ROC_AUC_SCORE)

print(f'Accuracy Score Logistic 2:', accuracy_score(y_test, y_pred))  # accuracy score = % of correct predictions

logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_decf)
# fpr = false positive rate, tpr = true positive rate
auc_logistic = auc(logistic_fpr, logistic_tpr)
logistic_auc_score = roc_auc_score(y_test, y_pred_decf)
print(f'Logistic (Iter Impute Method 2) AUC Score: {logistic_auc_score}')

plt.figure(figsize=(8, 6), dpi=100)
plt.title('ROC Curve Analysis', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.plot(logistic_fpr, logistic_tpr, label='Logistic Method 2 (AUC = %0.3f)' % auc_logistic, color=palette[1])
plt.legend()
plt.show()

"""
6. DECISION TREE MODEL
"""

# SPLIT
y = data_iter_imp.iloc[:, 0]
X = data_iter_imp.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# DECISION TREE MODEL
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred = model_dt.predict(X_test)
y_pred_probs = model_dt.predict_proba(X_test)  # NOTE: output = 2dim array
# NOTE: column 0 = P(x=0), column 1 = P(x=1)

# TESTING (ACCURACY, ROC_CURVE AND ROC_AUC_SCORE)
print(f'Accuracy Score Decision Tree:', accuracy_score(y_test, y_pred))  # accuracy score = % of correct predictions
decision_fpr, decision_tpr, threshold = roc_curve(y_test, y_pred_probs[:, 1])
auc_decision = auc(decision_fpr, decision_tpr)  # (roc and auc only function with the positive class probabilities)
decision_auc_score = roc_auc_score(y_test, y_pred_probs[:, 1])  # again, only select P(x = positive class)
print(f'Decision Tree AUC Score: {decision_auc_score}')

plt.figure(figsize=(8, 6), dpi=100)
plt.title('ROC Curve Analysis', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.plot(decision_fpr, decision_tpr, label='Decision Tree(AUC = %0.3f)' % auc_decision, color=palette[2])
plt.legend()
plt.show()

"""
7. RANDOM FOREST MODEL
"""

# SPLIT
y = data_iter_imp.iloc[:, 0]
X = data_iter_imp.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# RANDOM FOREST CLASSIFIER
model_rf = RandomForestClassifier(n_estimators=400, random_state=0)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
y_pred_probs = model_rf.predict_proba(X_test)[:, 1]  # output = prob(x=1)

# TESTING (ACCURACY, ROC_CURVE AND ROC_AUC_SCORE)
print(f'Accuracy Score Random Forest:', accuracy_score(y_test, y_pred))
random_forest_fpr, random_forest_tpr, threshold = roc_curve(y_test, y_pred_probs)
auc_random_forest = auc(random_forest_fpr, random_forest_tpr)
random_forest_auc_score = roc_auc_score(y_test, y_pred_probs)
print(f'Random Forest AUC Score: {random_forest_auc_score}')

# PLOTTING
plt.figure(figsize=(8, 6), dpi=100)
plt.title('ROC Curve Analysis', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.plot(random_forest_fpr, random_forest_tpr, label='Random Forest (AUC = %0.3f)' % auc_random_forest, color=palette[3])
plt.legend()
plt.show()

"""
8. BAGGING CLASSIFIER - TESTING N-ESTIMATORS
"""
#
# # SPLIT
# y = data_iter_imp.iloc[:, 0]
# X = data_iter_imp.iloc[:, 1:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# # create range = ensemble size to compare scores
# estimator_range = np.arange(60, 500, 20)
# models = []
# scores = []
#
# for n_estimators in estimator_range:
#     clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)
#     clf.fit(X_train, y_train)
#     models.append(clf)
#     scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))
#
# # Accuracy score vs n estimators
# plt.figure(figsize=(8, 6), dpi=100)
# plt.plot(estimator_range, scores, color=palette[6])
# plt.title('Accuracy vs N-Estimators', fontsize=16, fontweight='bold')
# plt.xlabel("n_estimators", fontsize=13, fontweight='bold',labelpad=12)
# plt.ylabel("score", fontsize=13, fontweight='bold',labelpad=12)
# plt.show()

"""
SEGMENT COMMENTED FOR RUNTIME
"""

"""
9. BAGGING CLASSIFIER MODEL
"""

# SPLIT
y = data_iter_imp.iloc[:, 0]
X = data_iter_imp.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# BAGGING CLASSIFIER MODEL
model_bag = BaggingClassifier(n_estimators=400)
model_bag.fit(X_train, y_train)
y_pred = model_bag.predict(X_test)
y_pred_probs = model_bag.predict_proba(X_test)  # output = 2-dim array
# Column 1 = Prob(x=0) Column 2 = Prob(x=1)


# TESTING (ACCURACY, ROC_CURVE AND ROC_AUC_SCORE)
print(f'Accuracy Score Bagging:', accuracy_score(y_test, y_pred))  # accuracy score = % of correct predictions
bagging_fpr, bagging_tpr, threshold = roc_curve(y_test, y_pred_probs[:, 1])  # select P(x = positive class)
auc_bagging = auc(bagging_fpr, bagging_tpr)  # (roc and auc only function with the positive class probabilities)
bagging_auc_score = roc_auc_score(y_test, y_pred_probs[:, 1])  # again, only select P(x = positive class)
print(f'Bagging Classifier AUC Score: {bagging_auc_score}')

plt.figure(figsize=(8, 6), dpi=100)
plt.title('ROC Curve Analysis', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.plot(bagging_fpr, bagging_tpr, label='Bagging Classifier(AUC = %0.3f)' % auc_bagging, color=palette[4])


plt.legend()
plt.show()


"""
10. FINAL COMPARISON
"""
plt.figure(figsize=(8, 6), dpi=100)
plt.title('ROC Curve Analysis', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold',labelpad=12)
plt.plot(bagging_fpr, bagging_tpr, label='Bagging Classifier(AUC = %0.3f)' % auc_bagging, color=palette[4])
plt.plot(random_forest_fpr, random_forest_tpr, label='Random Forest (AUC = %0.3f)' % auc_random_forest, color=palette[3])
plt.plot(decision_fpr, decision_tpr, label='Decision Tree(AUC = %0.3f)' % auc_decision, color=palette[2])
plt.plot(logistic_fpr, logistic_tpr, label='Logistic Method 2 (AUC = %0.3f)' % auc_logistic, color=palette[1])
plt.legend()
plt.show()




"""
11. FINAL PREDICTIONS USING RANDOM FOREST
"""

# ITER IMPUTE ON TARGET INPUT DATA
df_predictions = df_predictions.drop(columns="Unnamed: 0")
df_predictions.columns = df_predictions.columns.str.lower()
iter_imp = IterativeImputer(max_iter=20, random_state=0)  # initializing IterativeImputer
iter_imp.fit(df_predictions)
imp_df = iter_imp.transform(df_predictions)
df_predictions_imputed = pd.DataFrame(imp_df, columns=df_predictions.columns)

y_pred_final = model_rf.predict_proba(df_predictions_imputed)[:, 1]

submission_df = pd.DataFrame({
    "Id": range(len(y_pred_final)),
    "Probability": y_pred_final
})

submission_df.to_csv(r'C:\Users\koko\Desktop\ML\Credit_24\Submissions\final_predictions.csv', index=False)
