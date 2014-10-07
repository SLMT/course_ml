
clear all; clc;

load X.dat
load y.dat

% initailze variables
n = size(X, 1); % number of data points
d = size(X, 2); % number of fetures
expandX = [ones(n, 1), X]; % add ones column
w = zeros(d + 1, 1); % initailize w to zero vector

% prepare for descent
learningRate = 1;
lastEmp = model.regressor.LinearRegressor.calculateEMP(w, X, y);
times = 0;
slowDown = 0;
SLOW_DOWN_THRESHOLD = 1;

% iteration
while 1
    % gradient descent
    sum = 0;
    for i = 1:n
        sum = sum + (y(i) - w' * X(i,:)') * X(i,:)';
    end
    gradient = -2 * sum;
    newW = w - learningRate * gradient;
    
    % count emperical error
    emp = model.regressor.LinearRegressor.calculateEMP(newW, X, y);

    % check if emp descent slow down
    if emp < lastEmp && lastEmp - emp < SLOW_DOWN_THRESHOLD
        % 理論上上次 slowDown 應該就會更改 learning rate
        % 如果更改有變好，就不會進入上面的 if
        % 如果沒有變好，那就會進來這裡面，代表已經很難有進展了
        if (slowDown == 1) 
        	break;
        end
        
        slowDown = 1;
    end
    
    % check if this newW gives a better emp
    if emp >= lastEmp || slowDown == 1
        learningRate = learningRate / 10;
    else
        % update variables
        lastEmp = emp;
        w = newW;
        
        % reset slowDown variable (important)
        slowDown = 0;
    end
    
    times = times + 1;
end
times
emp
w

wLen = length(w);
predictedValue = w(2:wLen)' * oX + w(1);

%%% plot data %%%
scatter (oX, y,'g');
hold on;

%--- TODO: plot the regressor you train ---%
plot(oX, predictedValue, 'r');
hold off;


%{
newX = [ones(size(X)), X];
w = newX \ y;

a = linspace(min(X), max(X));
b = w(2) * a + w(1);

plot(X, y, '*', a, b, '-');
%}