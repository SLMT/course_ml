
clear all; clc;

load data/X
load data/y

obj = model.classify.LapRLSClassifier.train(X, y);
y2 = obj.predict(X);

plot_result(X, y2);