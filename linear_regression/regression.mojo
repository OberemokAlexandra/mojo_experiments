# не забыть добавлять эти переменные окружения при каждом запуске
# export MODULAR_HOME="$HOME/.modular"
# export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"

from python import Python
fn get_data() raises -> PythonObject:
    # This is equivalent to Python's `import numpy as np`
    
    let datasets = Python.import_module("sklearn.datasets"
        )   
     # Create matrix with numpy:
    let ds = datasets.load_diabetes()
    #let ds = datasets.make_regression()
    # попробовать разные штуки
    return ds

fn main() raises:
    # Create matrix 9 rows and 5 cols:
    let ds = get_data()
    let np = Python.import_module("numpy")
    var X = ds['data']
    var y = ds['target']
    #let X = ds[0]
    #let y = ds[1]
    # train_test split
    let model_selection = Python.import_module("sklearn.model_selection")
    let split = model_selection.train_test_split(X, y)
    let X_train = split[0]
    let X_test = split[1]
    let y_train = split[2]
    let y_test = split[3]
    # моделирование
    let linear_model = Python.import_module("sklearn.linear_model")
    var regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    let pred = regr.predict(X_test)
    # метрики
    let metrics = Python.import_module("sklearn.metrics")
    print("mean")
    print(np.mean(y_test))
    print("mse")
    print(metrics.mean_squared_error(y_test, pred))
    print("mae")
    print(metrics.mean_absolute_error(y_test, pred))
    print("r2")
    print(metrics.r2_score(y_test, pred))
    # print("Ridge Regression")
    # var r_regr = linear_model.Ridge()
    # r_regr.fit(X_train, y_train)
    # let r_pred = r_regr.predict(X_test)
    # # метрики
    # print("mse")
    # print(metrics.mean_squared_error(y_test, r_pred))
    # print("mae")
    # print(metrics.mean_absolute_error(y_test, r_pred))
    # print("r2")
    # print(metrics.r2_score(y_test, r_pred))
    

