import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from xgboost import *
from catboost import *

def use_columns():
    return {
                "ExternalRiskEstimate": 1,
                "MSinceOldestTradeOpen": 1,
                "MSinceMostRecentTradeOpen": 1,
                "AverageMInFile": 1,
                "NumSatisfactoryTrades": 1,
                "NumTrades60Ever2DerogPubRec": 1,
                "NumTrades90Ever2DerogPubRec": 1,
                "PercentTradesNeverDelq": 1,
                "MSinceMostRecentDelq": 1,
                "MaxDelq2PublicRecLast12M": 1,
                "MaxDelqEver": 1,
                "NumTotalTrades": 1,
                "NumTradesOpeninLast12M": 1,
                "PercentInstallTrades": 1,
                "MSinceMostRecentInqexcl7days": 1,
                "NumInqLast6M": 1,
                "NumInqLast6Mexcl7days": 1,
                "NetFractionRevolvingBurden": 1,
                "NetFractionInstallBurden": 1,
                "NumRevolvingTradesWBalance": 1,
                "NumInstallTradesWBalance": 1,
                "NumBank2NatlTradesWHighUtilization": 1,
                "PercentTradesWBalance": 1,
    }

def avg(model, x, y, eq):
    pred = model.predict(x)
    acc = [1 if eq(e, s) else 0 for (e,s) in zip(pred, y)]
    return np.average(acc)

def filter_diff(x):
    rp = x['RiskPerformance']
    dis = len(set(rp))
    return dis == 1

def get_data():
    read = pd.read_csv("heloc_dataset_v1.csv")
    data_columns = [f for f in read.columns if f != 'RiskPerformance']
    grouped = read.groupby(data_columns)
    filtered = grouped.filter(lambda x: True)
    filtered = filtered.sample(frac=1).reset_index(drop=True)
    return filtered['RiskPerformance'], filtered[data_columns]



def prep_data():
    res, data = get_data()
    res = [1 if t == "Good" else 0 for t in res]
    should_use = use_columns()
    data = data[[c for c in data.columns if should_use[c] == 1]]
    # for c in ["MaxDelq2PublicRecLast12M","MaxDelqEver"]:
    #      lb = LabelBinarizer()
    #      labeled = pd.DataFrame(lb.fit_transform(data[c]), columns=lb.classes_)
    #      for cl in lb.classes_:
    #          data[c + str(cl)] = labeled[cl]
    #      data = data[[col for col in data.columns if col != c]]
    missing = 0

    for c in data.columns:
          data[c] = [-9 if t == -7 or t == -8 or t == -9 else t for t in data[c]]
    #imp = SimpleImputer(missing_values=np.nan, strategy='median')
    orig_data = data
    #data = imp.fit_transform(data)

    return orig_data, np.array(res), pd.DataFrame(data, columns=orig_data.columns) #MinMaxScaler().fit_transform(data)


def monotonic():
    return {
"ExternalRiskEstimate" : 	1,
"MSinceOldestTradeOpen" : 	1,
"MSinceMostRecentTradeOpen" : 	1,
"AverageMInFile" : 	1,
"NumSatisfactoryTrades" : 	1,
"NumTrades60Ever2DerogPubRec" : -1,
"NumTrades90Ever2DerogPubRec" : -1,
"PercentTradesNeverDelq" : 	1,
"MSinceMostRecentDelq" : 	1,
"MaxDelq2PublicRecLast12M" : 	0,
"MaxDelqEver" : 	0,
"NumTotalTrades" : 	0,
"NumTradesOpeninLast12M" : -1,
"PercentInstallTrades" : 	0,
"MSinceMostRecentInqexcl7days" : 	1,
"NumInqLast6M" : -1,
"NumInqLast6Mexcl7days" : -1,
"NetFractionRevolvingBurden" : -1,
"NetFractionInstallBurden" : -1,
"NumRevolvingTradesWBalance" : 	0,
"NumInstallTradesWBalance" : 	0,
"NumBank2NatlTradesWHighUtilization" : -1,
"PercentTradesWBalance" : 	0
}

def get_model():
    #xgb = XGBClassifier(verbose=0, iterations=500, learning_rate=0.05, subsample=0.8)
    xgb = XGBRFClassifier()#monotone_constraints=tuple([mon_def[c] if c in mon_def else 0 for c in orig.columns]))
    xgb.missing = -9
    xgb.reg_lambda = 10
    xgb.reg_alpha = 10
    xgb.colsample_bytree = 0.5
    # xgb.max_leaf_nodes = 40
    xgb.max_depth = 8
    # xgb.objective = 'binary:logistic'
    return xgb


def find_best_params():
    orig, res, _ = prep_data()
    best_columns = ['ExternalRiskEstimate', 'NumSatisfactoryTrades', 'MSinceMostRecentInqexcl7days', 'NumBank2NatlTradesWHighUtilization', 'AverageMInFile', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumInqLast6M', 'NumTotalTrades', 'NetFractionRevolvingBurden', 'NumRevolvingTradesWBalance', 'NumTrades60Ever2DerogPubRec']
    orig = orig[best_columns]
    params = {
        'n_estimators': [10, 25, 50, 100, 150, 200, 500],
        'reg_lambda': [1e-5, 1, 2, 5, 10],
        #'reg_alpha': [1e-5, 1, 2, 5, 10],
        #'colsample_bytree': [0.5, 0.75, 1],
        'max_depth': [3, 5, 8, 11, 15],
        'learning_rate': [0.05, 0.1, 0.25, 0.5, 1],
        'verbose': [0],
        'subsample': [0.8]
    }
    xgb = CatBoostClassifier()

    rnd = RandomizedSearchCV(
        xgb,
        param_distributions=params,
        n_jobs=-1,
        n_iter=50,
        scoring=lambda a, b, c: avg(a, b, c, lambda x, y: int(round(x)) == y)
    )
    search = rnd.fit(orig, res)
    print("Best params: " + str(search.best_params_))
    print("Best score: " + str(search.best_score_))

def analyze():
    orig, res, data = prep_data()

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, res, train_size=0.8)
    mon_def = monotonic()

    xgb = get_model()
    xgb.monotone_constraints = tuple([mon_def[c] for c in data.columns])
    xgb.fit(Xtrain, Ytrain)
    print("Avg: " + str(avg(xgb, Xtest, Ytest, lambda x, y: int(round(x)) == y)))
    print("Avg on train: " + str(avg(xgb, Xtrain, Ytrain, lambda x, y: int(round(x)) == y)))
    roc_auc = roc_auc_score(Ytest, xgb.predict_proba(Xtest)[:, 1])
    print("Roc auc: " + str(roc_auc))

    rf = RandomForestClassifier(100, max_samples=1500, n_jobs=-1, max_features=5)
    rf.fit(Xtrain, Ytrain)
    print("Avg rf: " + str(avg(rf, Xtest, Ytest, lambda x, y: x == y)))
    print("Avg rf on train: " + str(avg(rf, Xtrain, Ytrain, lambda x, y: x == y)))
    roc_auc = roc_auc_score(Ytest, rf.predict_proba(Xtest)[:, 1])
    print("Roc auc rf: " + str(roc_auc))


if __name__ == "__main__":
    analyze()