%%% the dummy regressor predicts the value by mean value of y

% handle �N��o�� class �O pass by reference
classdef LinearRegressor < handle
	properties
        w; % training ��o�쪺���u�Ѽ�
    end
   
	methods
        % constructor
        function linearRegressorObj = LinearRegressor (w)
            linearRegressorObj.w = w;
        end
       
        % �� w �� predict �� method
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
        
        % �����k Training
        function linearRegressorObj = leftDivisionTrain (X, y)
            % �b�Ĥ@�� column ������J 1
            newX = [ones(size(X)), X];
            % �����k (�̤p����k)
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
                % �o��ھ� emp ���T�إi�બ�p
                % 1. �۸� lastEmp �����ϤɩΤ��� (�S���i)
                if emp >= lastEmp
                    % ���լݬݭ��C learning rate �|���|�ܦn
                    learningRate = learningRate / 10;
                    
                    lastCase = 1;
                % 2. �۸� lastEmp �U���Ӥ� (�i�B�Ӥ�)
                elseif lastEmp - emp < SLOW_DOWN_THRESHOLD
                    % �W�����`�A�o���ܺC
                    if lastCase == 3
                        % ���լݬݴ��� learning rate �|���|�ܦn
                        learningRate = learningRate * 10;
                        
                    % �W���N�ܺC�A�o���٬O�C�A�Ϊ̤W���W�L�A�o���S�ܺC�A�i�H�����F
                    elseif lastCase == 2 || lastCase == 1
                        break;
                    end

                    lastCase = 2;
                % 3. �@�����`
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