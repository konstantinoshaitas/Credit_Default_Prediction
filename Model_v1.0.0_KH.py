#PACKAGE IMPORT
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.metrics import classification_report


#STYLE FOR PLOTTING GRAPHS

sns.set_style("whitegrid")
sns.set_context("paper", font_scale = 1.0)


#IMPORT DATA

df_og = pd.read_csv(r"C:\Users\koko\Desktop\ML\Credit_24\data\train.csv")
df_og = df_og.drop(columns="Unnamed: 0")
df_og.columns = df_og.columns.str.lower() #Main DataFrame


#DATA PREPROCESSING

df = df_og.dropna() #Drop Null Values - Main Dataframe to work from



