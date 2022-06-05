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

def kernel_importance(X, y, kernel): 
    krr = KernelRidge(alpha=1.0, kernel = kernel)
    feature_names = ["tight", "leg", "foot"]
    krr.fit(X, y)
    
    return sorted(list(zip(feature_names, X.T @ krr.dual_coef_)))

def linear_models(X, y): 
    df = pd.DataFrame(
        data = np.hstack((X, y.reshape(-1,1)))
    )
    df.columns = ["tight", "leg", "foot", "AvgTotalReward"]

    complete_model = sfm.ols("AvgTotalReward ~ tight + leg + foot", data = df).fit()
    anova = sm.stats.anova_lm(complete_model)
    return anova

def main(): 
    data = "variant/dynamics.txt"
    df = load_data(data)
    
    # X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X, y = 10*np.random.random(size = (100, 3)), 3*np.random.random(size = (100,))

    rf_imp = random_forest_importance(X, y)
    ker_imp = kernel_importance(X, y, "rbf")
    anova_test = linear_models(X, y)

    print(rf_imp)
    print(ker_imp)
    print(anova_test)

if __name__ == "__main__": 
    main()