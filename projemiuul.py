from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.metrics import classification_report, roc_auc_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.set_option('display.width', 500)

df = pd.read_csv("miuul proje/Data.csv", sep=",")
df.head()
df.info()
df.isnull().sum()
df.columns
df.describe().T
df.shape






#Özellik Mühendisliği ! her bir oyuncunun toplam gol süre  şut  değerleri bulunmalı ve maç basına olarak bölünmeli

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


for col in num_cols:
    print(col, check_outlier(df, col))

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

for col in num_cols:
    df = remove_outlier(df, col)






tf =df.groupby("Player Names").agg({"Matches_Played" : "sum",
                                    "Mins" : "sum","Goals":"sum", "xG": "mean","xG Per Avg Match" : "mean",
                                    "Shots":"sum","OnTarget":"sum","Shots Per Avg Match": "mean","On Target Per Avg Match":"mean"})
tf["Player Names"] = tf.index
tf.index = tf["Player Names"]
tf =tf.reset_index(drop=True)
tf.head()
tf.shape

tf["Goal Per Game"] = tf["Goals"]/tf["Matches_Played"]
tf["Mins Per Game"] = tf["Mins"]/tf["Matches_Played"]
tf["Goal Per Mins"] = tf["Goals"]/tf["Mins"]


cat_cols, num_cols, cat_but_car = grab_col_names(tf)

for col in num_cols:
    print(col, check_outlier(tf, col))

for col in num_cols:
    print(col, remove_outlier(tf, col))

tf.shape


X = tf.drop(["Goal Per Game","Player Names",],axis=1)
y = tf["Goal Per Game"]

tf.shape



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

model = LinearRegression().fit(X,y)
model.intercept_ # -0.6904684494975869
model.coef_
y_pred =model.predict(X)
tf["tahmini_gol"] = y_pred
mean_squared_error(y,y_pred) # 0.0003609985926437382

#train
y_pred_train = model.predict(X_train)
mean_squared_error(y_train, y_pred_train)# 0.00023536602219508614
#test
y_pred_test = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred_test))#0.02930849600639838
#10 katli cros-validation
np.mean(np.sqrt(-cross_val_score(model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error"))) #0.018235866433024123

g = sns.regplot(x=tf["Player Names"].head(10), y=tf["tahmini_gol"].head(10), scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")
g.set_ylabel("maç başına gol")
g.set_xlabel("ortalama şut")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

#Sınıflandırma
tf["tahmini_gol"].describe().T
tf["sınıf"] = tf["tahmini_gol"].apply(lambda x : 1 if x > tf["tahmini_gol"].mean() else 0) # 1 hucumcu 0 defans



b = tf["sınıf"]


def base_models(X, y, scoring="roc_auc",verbose = False):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose=-1),),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")




base_models(X,b)


cart_model = DecisionTreeClassifier(random_state=1).fit(X, b)

X_train, X_test, b_train, b_test = train_test_split(X, b, test_size=0.30, random_state=45)






# Train Hatası
tahmini_sınıf_train = cart_model.predict(X_train)

print(classification_report(b_train, tahmini_sınıf_train))
roc_auc_score(b_train, tahmini_sınıf_train)

# Test Hatası
tahmini_sınıf_test = cart_model.predict(X_test)
tahmini_sınıf_test= cart_model.predict_proba(X_test)[:, 1]
print(classification_report(b_test,tahmini_sınıf_test ))
roc_auc_score(b_test, tahmini_sınıf_test)
