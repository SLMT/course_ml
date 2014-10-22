
load data/training/10/X.dat
load data/training/10/y.dat

phiX = [X.^2, X];
num = size(phiX, 1);
dimension = size(phiX, 2);

% adjustable variables
learningRate = 0.000001;
epi = 0.000001;

% Initialize variables
w = [0.9487; -3.3650];
b = 1;
updateVector = ones(dimension, 1);

% converage
while norm(updateVector) > epi
    % predict
    predicted = sign(phiX * w + b);
    
    % update w
    updateVector = learningRate * ((y - predicted)' * phiX)';
    w = w + updateVector;
    
    % update b
    b = b + sum(y - predicted);
end

% draw
lineX = linspace(min(X), max(X));
comX = [lineX' .^ 2, lineX']';
lineY = w' * comX + b;
plot(lineX, lineY);
hold on;

for t = 1 : num
    if y(t) == 1
        plot(X(t), 0, 'ro');
    else
        plot(X(t), 0, 'go');
    end
end

hold off;