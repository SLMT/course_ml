import model.classifier.PerceptronClassifier

clear; % clear variables

% ========================================================

% load training data set
load data/training/10/X.dat
load data/training/10/y.dat

% training
classifier1 = PerceptronClassifier.train(X, y, 3, 0.1, 0.000001);

% ========================================================

% load training data set
load data/training/100/X.dat
load data/training/100/y.dat

% training
classifier2 = PerceptronClassifier.train(X, y, 3, 0.0001, 0.00001);

% ========================================================

% load training data set
load data/training/1000/X.dat
load data/training/1000/y.dat

% training
classifier3 = PerceptronClassifier.train(X, y, 3, 0.1, 0.0001);
drawResult(classifier3.param, X, y);

% ========================================================

% load testing data set
load data/testing/X.dat
load data/testing/y.dat

% testing
predictedValue1 = classifier1.predict(X);
predictedValue2 = classifier2.predict(X);
predictedValue3 = classifier3.predict(X);

% count generalization error
num = size(X, 1);
error = [0, 0, 0];

for i = 1 : num
    if (y(i) ~= predictedValue1(i))
        error(1) = error(1) + 1;
    end
    if (y(i) ~= predictedValue2(i))
        error(2) = error(2) + 1;
    end
    if (y(i) ~= predictedValue3(i))
        error(3) = error(3) + 1;
    end
end

error = error / num

% ========================================================

% draw gereralization error
lineX = [10, 100, 1000];
lineY = error;
plot(lineX, lineY);

hold on;

% count consistency bound
optError = 0;
VC = 4;
bound = [0, 0, 0];
for i = 1 : 3
    N = lineX(i);
    bound(i) = 2 * sqrt(32 * ((VC * log10(N * exp(1) / VC)) + log10(4/0.1)) / N);
end
plot(lineX, bound);

hold off;
