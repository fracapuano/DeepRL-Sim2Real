import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
import statsmodels.api as sm
import statsmodels.formula.api as sfm

def load_data(path): 
    df = pd.read_csv(path, delimiter = " ")
    return df

def random_forest_importance(X, y): 
    rs = 42
    reg = RandomForestRegressor(n_jobs = -1, random_state = rs, verbose = 1)
    reg.fit(X, y)
    parts = ["tight", "leg", "foot"]
    
    return sorted(list(zip(parts, reg.feature_importances_)))

def linear_models(X, y): 
    df = pd.DataFrame(
        data = np.hstack((X, y.reshape(-1,1)))
    )
    df.columns = ["tight", "leg", "foot", "AvgTotalReward"]

    complete_model = sfm.ols("AvgTotalReward ~ tight + leg + foot", data = df).fit()
    return complete_model.summary()

def main(): 
    data = "dynamics.txt"
    df = load_data(data)
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    
    rf_imp = random_forest_importance(X, y)
    t_test = linear_models(X, y)

    print(rf_imp)
    print(t_test)

if __name__ == "__main__": 
    main()
