from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

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

