import model.classify.SMOClassifier

clear %clear workspace

% some constants
epsilon = 0.000001;

% import data set
load data/X.dat
load data/y.dat

% training
myClassifier = SMOClassifier.train(X, y, 1);
label = myClassifier.predict(X);

% get arguments
data = myClassifier.data;

% plot data
scatter (X(y==1,1),X(y==1,2),'g');
hold on;
scatter (X(y==-1,1),X(y==-1,2),'b');

% plot support verters
plot(data(:, 1), data(:, 2), 'kx');

% show error rate
errorRate = size(label(label ~= y), 1) / size(label, 1)

hold off;
