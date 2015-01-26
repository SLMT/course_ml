import model.classify.MLFinalClassifier

clear %clear workspace

% import data set
load data/X.mat
load data/y.mat
load data/Xtest.mat
load data/ytest.mat

% Z-Normalization
n = size(X, 1);
X_mean = repmat( mean(X), [n, 1] );
X_std =repmat( std(X), [n, 1] );
X = (X - X_mean) ./ X_std;

% get labeled data
labeled_x = X(y ~= 0, :);
labeled_y = y(y ~= 0);

% LDA
w = LDA(labeled_x, labeled_y);
lda_X = [ones(n,1) X] * w';

test_n = size(Xtest, 1);
lda_X_test =  [ones(test_n,1) Xtest] * w';

% training
myClassifier = MLFinalClassifier.train(lda_X, y);
label = myClassifier.predict(lda_X_test);

% show error rate
accuracy = 1 - size(label(label ~= ytest), 1) / size(label, 1)

