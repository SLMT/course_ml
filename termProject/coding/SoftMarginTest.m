import model.classify.SoftMarginLinearClassifier

clear %clear workspace

% import data set
load data/X.dat
load data/y.dat

% training
myClassifier = SoftMarginLinearClassifier.train(X,y);
label = myClassifier.predict(X);

% get arguments
w = myClassifier.w;
b = myClassifier.b;
xi = myClassifier.xi;

% plot data
scatter (X(y==1,1),X(y==1,2),'g');
hold on;
scatter (X(y==-1,1),X(y==-1,2),'b');

% plot support verters
plot(X(xi > 0.1,1), X(xi > 0.1, 2), 'kx');

% plot boundary
lineA = -w(1) / w(2);
lineB = b / w(2);
lineX = linspace(-60, 20);
lineY = lineX * lineA + lineB;
plot(lineX, lineY, 'k');

lineY = lineY + 1 / w(2);
plot(lineX, lineY, 'k--');

lineY = lineY - 2 / w(2);
plot(lineX, lineY, 'k--');

% show error rate
errorRate = size(label(label ~= y), 1) / size(label, 1)
numOfSupportVec = size(myClassifier.xi(myClassifier.xi > 0.000001), 1)

hold off;
