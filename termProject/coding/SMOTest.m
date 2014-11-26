import model.classify.SMOClassifier

clear %clear workspace

% import data set
load data/X.dat
load data/y.dat

% training
myClassifier = SMOClassifier.train(X,y);

% plot data
scatter (X(y==1,1),X(y==1,2),'g');
hold on;
scatter (X(y==-1,1),X(y==-1,2),'b');

% plot boundary


hold off;
