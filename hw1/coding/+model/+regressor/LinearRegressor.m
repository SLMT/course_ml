%%% the dummy regressor predicts the value by mean value of y

% handle 硂 class 琌 pass by reference
classdef LinearRegressor < handle
	properties
        w; % training 眔絬把计
    end
   
	methods
        % constructor
        function linearRegressorObj = LinearRegressor (w)
            linearRegressorObj.w = w;
        end
       
        % ノ w ㄓ predict  method
        function predictedValue = predict (obj, X)
            wLen = length(obj.w);
            predictedValue = (obj.w(2:wLen))' * X + obj.w(1);
        end
    end
    
    % static methods
	methods (Static)
        function linearRegressorObj = train (X, y)
            % create object
            linearRegressorObj = model.regressor.LinearRegressor.cvxTrain(X, y);
        end
        
        % オ埃猭 Training
        function linearRegressorObj = leftDivisionTrain (X, y)
            % 材 column 场 1
            newX = [ones(size(X)), X];
            % オ埃猭 (程キよ猭)
            param = newX \ y;
            
            % create object
            linearRegressorObj = model.regressor.LinearRegressor(param);
        end
        
        % CVX Training
        function linearRegressorObj = cvxTrain (X, y)
        	% initailze variables
            n = size(X, 1); % number of data points
            d = size(X, 2); % number of fetures
            expandX = [ones(n, 1), X]; % add ones column
            param = zeros(d + 1, 1); % initailize param to zero vector
            
            % cvx optimization
            cvx_begin
                variable param(d+1)
                minimize( norm(expandX * param - y))
            cvx_end

            % create object
            linearRegressorObj = model.regressor.LinearRegressor(param);
        end
        
        % calculate emperical error
        function emp = calculateEMP(w, X, y)
            % initialize variables
            n = size(X, 1);
            emp = 0;
            tmpX = [ones(n, 1), X];
            
            % sum all square error
            for i = 1:n
                innerPart = y(i) - w' * tmpX(i,:)';
                emp = emp + innerPart * innerPart;
            end
        end
	end
end