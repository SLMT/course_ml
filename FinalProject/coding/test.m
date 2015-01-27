import model.classify.MLFinalClassifier

clear; close; clc;

%% import data set
load data/X.mat
load data/y.mat
load data/Xtest.mat
load data/ytest.mat

%% Training
myClassifier = MLFinalClassifier.train(X, y);

%% Make prediction
label = myClassifier.predict(Xtest);

%% show error rate
accuracy = 1 - size(label(label ~= ytest), 1) / size(label, 1);
fprintf( 'Testing Accuracy : %.2f%%\n', accuracy*100 );
