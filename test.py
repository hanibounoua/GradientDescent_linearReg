import numpy as np
from sklearn.model_selection import train_test_split
from linearRegression import linearRegression

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    X = points[:, 0]
    y = points[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 123)
    model = linearRegression(learningRate = .0001, n_iter = 1000)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    return {"Model": model, "Mean Squared Error": np.mean((y_predicted - y_test)**2)}

if __name__ == "main":
    run()
