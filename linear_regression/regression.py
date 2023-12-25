from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def get_data():
    """returns dataset for modeling"""
    return load_diabetes()


def main():
    """performs all the modeling actions"""
    ds = get_data()
    X, y = ds['data'], ds['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    pred = regr.predict(X_test)
    print("mean")
    print(np.mean(y_test))
    print("mse")
    print(mean_squared_error(y_test, pred))
    print("mae")
    print(mean_absolute_error(y_test, pred))
    print("r2")
    print(r2_score(y_test, pred))    


main()
