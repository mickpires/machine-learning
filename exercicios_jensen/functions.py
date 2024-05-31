from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np
from math import floor

def getRmse(X_train,y_train,X_test,y_test,degree):
    model = Pipeline([
        ('std_scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree)),
        ('lin_reg', LinearRegression())
    ])
    model.fit(X_train,y_train)
    rmse_train = (np.sqrt(mean_squared_error(y_train,model.predict(X_train))))
    rmse_test = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
    return rmse_train, rmse_test

def eqmByComplexity(X_train,y_train,X_test,y_test,degree):
    model = Pipeline([
        ('std_scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree)),
        ('lin_reg', LinearRegression())
    ])
    model.fit(X_train,y_train)


def KFold(x,y,k):
    k_fold_size = floor(len(y)/k)
    y_new = np.empty((k,k_fold_size))
    x_new = np.empty((k,k_fold_size))
    for j in range(k_fold_size):
        for i in range(k):
            index = np.random.randint(0,len(y))
            y_new[i,j] = y[index]
            x_new[i,j] = x[index]
            if len(y) > 0:
                y = np.delete(y,index)
                x = np.delete(x,index)
    return x_new,y_new