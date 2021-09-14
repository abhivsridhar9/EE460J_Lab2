import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
def q4():
    #Preprocessing
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    train.head()
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']))
    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
    #prices.hist()

    #log transform the target:
    train["SalePrice"] = np.log1p(train["SalePrice"])

    #log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = pd.get_dummies(all_data)

    #filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())

    #creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    #Models
    from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
    from sklearn.model_selection import cross_val_score

    def rmse_cv(model):
        rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
        return(rmse)





    #Part1
    ridge_regression = rmse_cv(Ridge(alpha= .1)).mean()
    print("a = .1 score:", ridge_regression)

    model_ridge = Ridge(10)
    model_ridge.fit(X_train, y)
    l2_pred = model_ridge.predict(X_train)
    #solution = pd.DataFrame({"id": test.Id, "SalePrice": np.expm1(l2_pred)})
    #solution.to_csv("Ridge_Regression.csv", index=False)
    #print("l2 pred ", np.expm1(l2_pred))

    #Ridge V Lasso CV
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean()
                for alpha in alphas]
    cv_ridge = pd.Series(cv_ridge, index=alphas)
    #print(cv_ridge)
    #cv_ridge.plot(title="Validation - Just Do It")
    #plt.xlabel("alpha")
    #plt.ylabel("rmse")
    #plt.show()
    print("Ridge Score:",cv_ridge.min())


    model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(X_train, y)
    l1_pred = model_lasso.predict(X_train)
    print(l1_pred)
    #print(test.Id)
    #solution = pd.DataFrame({"id": test.Id, "SalePrice": l1_pred})
    #solution.to_csv("Lasso_Regression.csv", index=False)
    #print("l1 pred ", np.expm1(l1_pred))
    print("Lasso Score",rmse_cv(model_lasso).mean())


    #L0 Norm
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    l0_norm = []
    for alpha in alphas:
        model_lasso = LassoCV(alphas=[alpha]).fit(X_train, y)
        coef = pd.Series(model_lasso.coef_, index=X_train.columns)
        l0_norm.append(sum(coef!=0))
    plt.scatter(alphas, l0_norm)
    plt.title("L0 Norm v Alphas")
    plt.xlabel("Alpha")
    plt.ylabel("L0 Norm")
    plt.show()

    #print(coef)
    #print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
    #    sum(coef == 0)) + " variables")


    #ADDING OUTCOME AS FEATURE
    # Preprocessing
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    train.head()
    all_data = pd.concat((train.loc[:, 'MSSubClass':'SalePrice'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))
    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({"price": train["SalePrice"], "log(price + 1)": np.log1p(train["SalePrice"])})
    prices.hist()

    # log transform the target:
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = pd.get_dummies(all_data)

    # filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())

    # creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_train.insert(0, "l1_pred", l1_pred)
    X_train.insert(0, "l2_pred", l2_pred)
    #print(X_train.size, l1_pred.size, l2_pred.size)
    #print(X_train)
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    alphas = [.05, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean()
                for alpha in alphas]
    cv_ridge = pd.Series(cv_ridge, index=alphas)
    print("Ridge Regression w Models as Output Score:", cv_ridge.min())


    #XGBOOST
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error as MSE

    # Preprocessing
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    train.head()
    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))
    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({"price": train["SalePrice"], "log(price + 1)": np.log1p(train["SalePrice"])})
    prices.hist()

    # log transform the target:
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = pd.get_dummies(all_data)

    # filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())

    # creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice
    #print(X_train)
    dtrain = xgb.DMatrix(X_train, label=y)
    dtest = xgb.DMatrix(X_test)

    params = {"max_depth": 1, "eta": 0.1}
    model = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)

    model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)  # the params were tuned using xgb.cv
    model_xgb.fit(X_train, y)
    xgb_preds = np.expm1(model_xgb.predict(X_test))
    #print(len(xgb_preds))
    print(len(test.Id))

    #print("RMSE : % f" % (rmse))
    lasso_preds = np.expm1(model_lasso.predict(X_test))

    #predictions = pd.DataFrame({"xgb": xgb_preds, "lasso": lasso_preds})
    #predictions.plot(x="xgb", y="lasso", kind="scatter")
    #preds = 0.7 * lasso_preds + 0.3 * xgb_preds

    solution = pd.DataFrame({"id": test.Id, "SalePrice": xgb_preds})
    solution.to_csv("XGB_Regression.csv", index=False)

if __name__ == '__main__':
    q4()
