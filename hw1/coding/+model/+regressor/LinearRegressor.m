%%% the dummy regressor predicts the value by mean value of y

% handle 代表這個 class 是 pass by reference
classdef LinearRegressor < handle
	properties
        w; % training 後得到的直線參數
    end
   
	methods
        % constructor
        function linearRegressorObj = LinearRegressor (w)
            linearRegressorObj.w = w;
        end
       
        % 用 w 來 predict 的 method
        function predictedValue = predict (obj, X)
            wLen = length(obj.w);
            predictedValue = (obj.w(2:wLen))' * X + obj.w(1);
        end
    end
    
    % static methods
	methods (Static)
        function linearRegressorObj = train (X, y)
            
            % create object
            linearRegressorObj = model.regressor.LinearRegressor.leftDivisionTrain(X, y);
        end
        
        % 左除法 Training
        function linearRegressorObj = leftDivisionTrain (X, y)
            % 在第一個 column 全部放入 1
            newX = [ones(size(X)), X];
            % 左除法 (最小平方法)
            param = newX \ y;
            
            % create object
            linearRegressorObj = model.regressor.LinearRegressor(param);
        end
        
        % Gradient Descent Training
        function linearRegressorObj = gradientDescentTrain (X, y)
        	% initailze variables
            n = size(X, 1); % number of data points
            d = size(X, 2); % number of fetures
            expandX = [ones(n, 1), X]; % add ones column
            param = zeros(d + 1, 1); % initailize param to zero vector

            % prepare for descent
            learningRate = 1;
            lastEmp = model.regressor.LinearRegressor.calculateEMP(param, X, y);
            times = 0;
            lastCase = -1;
            SLOW_DOWN_THRESHOLD = 0.1;

            % iteration
            while 1
                % gradient descent
                sum = 0;
                for i = 1:n
                    sum = sum + (y(i) - param' * expandX(i,:)') * expandX(i,:)';
                end
                gradient = -2 * sum;
                newParam = param - learningRate * gradient;

                % count emperical error
                emp = model.regressor.LinearRegressor.calculateEMP(newParam, X, y);
                
                % update variables
                % 這邊根據 emp 有三種可能狀況
                % 1. 相較 lastEmp 不降反升或不變 (沒長進)
                if emp >= lastEmp
                    % 嘗試看看降低 learning rate 會不會變好
                    learningRate = learningRate / 10;
                    
                    lastCase = 1;
                % 2. 相較 lastEmp 下降太少 (進步太少)
                elseif lastEmp - emp < SLOW_DOWN_THRESHOLD
                    % 上次正常，這次變慢
                    if lastCase == 3
                        % 嘗試看看提升 learning rate 會不會變好
                        learningRate = learningRate * 10;
                        
                    % 上次就很慢，這次還是慢，或者上次超過，這次又很慢，可以結束了
                    elseif lastCase == 2 || lastCase == 1
                        break;
                    end

                    lastCase = 2;
                % 3. 一切正常
                else
                    lastEmp = emp;
                    param = newParam;
                    lastCase = 3;
                end
                
                times = times + 1;
            end

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