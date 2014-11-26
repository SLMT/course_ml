import model.classify.SMOClassifier

clear %clear workspace

% some constants
epsilon = 0.000001;

% import data set
load data/X.dat
load data/y.dat

% training
myClassifier = SMOClassifier.train(X,y);

% get arguments
sv = myClassifier.data;

% plot data
scatter (X(y==1,1),X(y==1,2),'g');
hold on;
scatter (X(y==-1,1),X(y==-1,2),'b');

% plot support verters
plot(sv(:, 1), sv(:, 2), 'kx');


hold off;
