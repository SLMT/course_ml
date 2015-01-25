import model.classify.MLFinalClassifier

clear %clear workspace

% import data set
load data/X.mat
load data/y.mat
load data/Xtest.mat
load data/ytest.mat

% training
myClassifier = MLFinalClassifier.train(X, y);
label = myClassifier.predict(Xtest);

% show error rate
accuracy = 1 - size(label(label ~= ytest), 1) / size(label, 1)

