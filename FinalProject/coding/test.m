import model.classify.MLFinalClassifier

clear; close; clc;

%% import data set
load data/X.mat
load data/y.mat
load data/Xtest.mat
load data/ytest.mat

%% Z-Normalization
n = size(X, 1);
ntest = size(Xtest, 1);
Xmean = mean(X);
Xstd = std(X);
X = (X - repmat(Xmean,[n,1])) ./ repmat(Xstd,[n,1]); %Repmat for Dimension agreeement
Xtest = (Xtest - repmat(Xmean,[ntest,1])) ./ repmat(Xstd,[ntest,1]);

%% get labeled data
labeled_x = X(y ~= 0, :);
labeled_y = y(y ~= 0);

%% LDA
w = LDA(labeled_x, labeled_y);
lda_X = [ones(n,1) X] * w';

test_n = size(Xtest, 1);
lda_X_test =  [ones(test_n,1) Xtest] * w';

%% Training
myClassifier = MLFinalClassifier.train(lda_X, y);

%% Make prediction
label = myClassifier.predict(lda_X_test);

%% show error rate
accuracy = 1 - size(label(label ~= ytest), 1) / size(label, 1);
fprintf( 'Accuracy : %.2f%%\n', accuracy*100 );
