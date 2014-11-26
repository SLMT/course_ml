import model.classify.SoftMarginLinearClassifier
import model.classify.SMOClassifier

clear %clear workspace

% import data set
load data/X.dat
load data/y.dat

% random sample
[set50, index] = datasample(X, 50);
y50 = y(index);
[set100, index] = datasample(X, 100);
y100 = y(index);
[set150, index] = datasample(X, 150);
y150 = y(index);
set200 = X;
y200 = y;

% Soft-Margin
tic
SoftMarginLinearClassifier.train(set50, y50);
toc

tic
SoftMarginLinearClassifier.train(set100, y100);
toc

tic
SoftMarginLinearClassifier.train(set150, y150);
toc

tic
SoftMarginLinearClassifier.train(set200, y200);
toc

% SMO, no cache
tic
SMOClassifier.train(set50, y50, 0);
toc

tic
SMOClassifier.train(set100, y100, 0);
toc

tic
SMOClassifier.train(set150, y150, 0);
toc

tic
SMOClassifier.train(set200, y200, 0);
toc

% SMO, with cache
tic
SMOClassifier.train(set50, y50, 1);
toc

tic
SMOClassifier.train(set100, y100, 1);
toc

tic
SMOClassifier.train(set150, y150, 1);
toc

tic
SMOClassifier.train(set200, y200, 1);
toc
