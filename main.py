# -*- coding: utf-8 -*-
from .logistic_regression import LogisticRegression


def load_dataset():
    # load iris dataset
    import pandas as pd
    dataset = pd.read_csv("./optimization_project/iris.data", sep=",", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "label"])
    X, Y = [], []
    for _, x in dataset[dataset["label"]!="Iris-virginica"].iterrows():
        X.append((x["sepal_length"], x["sepal_width"], x["petal_length"], x["petal_width"]))
        y = 1 if x["label"]=="Iris-setosa" else -1
        Y.append(y)
    return X, Y


def main():
    X, Y = load_dataset()
    model = LogisticRegression(X, Y)
    model.fit()
    correct = 0
    for x, y in zip(X, Y):
        pred = model.predict(x)
        pred = 1 if pred >= 0.5 else -1
        correct += 1 if pred == y else 0
    print("correct prediction rate: {0}%.".format(float(correct)/len(Y)*100))


if __name__ == "__main__":
    # print(load_dataset())
    main()