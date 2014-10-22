import model.classifier.PerceptronClassifier

clear; % clear variables

% load data set
load data/training/10/X.dat
load data/training/10/y.dat

% training
classifier = PerceptronClassifier.train(X, y, 2);

% draw
lineX = linspace(min(X), max(X));
lineY = classifier.param(3) * (lineX .^ 2) + classifier.param(2) * lineX + classifier.param(1);
plot(lineX, lineY);
%{
lineX = linspace(min(X), max(X));
comX = [lineX' .^ 2, lineX']';
lineY = w' * comX + b;
plot(lineX, lineY);
%}


hold on;

for t = 1 : size(X, 1)
    if y(t) == 1
        plot(X(t), 0, 'ro');
    else
        plot(X(t), 0, 'go');
    end
end

hold off;