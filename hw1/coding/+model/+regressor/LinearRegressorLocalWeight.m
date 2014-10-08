

classdef LinearRegressorLocalWeight < model.regressor.LinearRegressor
   properties
      X, y, tau; % data sets and weight parameter 'tau'
   end
   
   methods
       % constructor
       function regressorObj = LinearRegressorLocalWeight (X, y)  
           regressorObj@model.regressor.LinearRegressor(0);
           regressorObj.X = X;
           regressorObj.y = y;
       end
       
       % lazy learning predict
       function predictedValues = predict (obj, target_x, cfg)
            % initialize variables
            numData = size(target_x, 1);
            d = size(obj.X, 2); % number of fetures
            predictedValues = zeros(numData, 1);
            obj.tau = cfg('tau');

            % trainning
            for i = 1:numData
                param = cvxTrain(obj, target_x(i, :)');
                predictedValues(i) = target_x(i, :) * param(2:d + 1) + param(1);
            end
       end
       
       function param = cvxTrain(obj, target_x)
            % initailze variables
            n = size(obj.X, 1); % number of data points
            d = size(obj.X, 2); % number of fetures
            expandX = [ones(n, 1), obj.X]; % add ones column
            param = zeros(d + 1, 1); % initailize param to zero vector
            
            % construct weighted matrix L
            L = zeros(n, n);
            for i = 1:n
                upper = (target_x - obj.X(i,:)') ^ 2;
                lower = -2 * (obj.tau) ^ 2;
                L(i,i) = exp(upper / lower);
            end
            
            % cvx optimization
            cvx_begin
                variable param(d+1)
                minimize( (expandX * param - obj.y)' * L * (expandX * param - obj.y))
            cvx_end
       end
   end
   
   methods (Static)
      function regressorObj = train (X, y)
        regressorObj = model.regressor.LinearRegressorLocalWeight(X, y);
      end
   end
end