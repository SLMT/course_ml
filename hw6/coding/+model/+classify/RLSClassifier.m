classdef RLSClassifier
    
    properties
        alpha, kernel;
    end
    
    methods
        function obj = RLSClassifier(alpha, kernel)
            obj.alpha = alpha;
            obj.kernel = kernel;
        end
        
        function y = predict(obj, X)
            y = obj.kernel * obj.alpha;
            y = sign(y);
        end
    end
    
    methods (Static)
        function classifierObj = train(X, y)
            % Define some constants
            n = size(X, 1);
            gama = 10;
            lamda = 1;
            
            % Compute K (Gaussian Kernel)
            K = zeros(n);
            for i = 1 : n
                for j = i : n
                    result = exp(- gama * norm(X(i, :) - X(j, :)) ^ 2);
                    
                    % fill in result
                    K(i, j) = result;
                    K(j, i) = result;
                end
            end
            
            % Compute A (To eliminate unlabeled data)
            A = zeros(n);
            for i = 1 : n
                if y(i) ~= 0
                    A(i, i) = 1;
                end
            end

            % Use CVX to sole Objective
            cvx_begin
                variable c(n)
                minimize((y - K * c)' * A * (y - K * c) + lamda * c' * K * c)
            cvx_end
            
            % Construct object
            classifierObj = model.classify.RLSClassifier(c, K);
        end
    end
    
end

