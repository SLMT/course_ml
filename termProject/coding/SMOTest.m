import model.classify.SMOClassifier

clear %clear workspace

% some constants
epsilon = 0.000001;

% import data set
load data/X.dat
load data/y.dat

% training
myClassifier = SMOClassifier.train(X,y);
label = myClassifier.predict(X);

% get arguments
w = myClassifier.w;
b = myClassifier.b;
alpha = myClassifier.alpha;

% plot data
scatter (X(y==1,1),X(y==1,2),'g');
hold on;
scatter (X(y==-1,1),X(y==-1,2),'b');

% plot support verters
plot(X(alpha > epsilon,1), X(alpha > epsilon, 2), 'kx');

% plot boundary
lineA = -w(1) / w(2);
lineB = -b / w(2);
lineX = linspace(-60, 20);
lineY = lineX * lineA + lineB;
plot(lineX, lineY, 'k');

lineY = lineY + 1 / w(2);
plot(lineX, lineY, 'k--');

lineY = lineY - 2 / w(2);
plot(lineX, lineY, 'k--');

% show error rate
errorRate = size(label(label ~= y), 1) / size(label, 1)
numOfSupportVec = size(alpha(alpha > 0.000001), 1)

hold off;
