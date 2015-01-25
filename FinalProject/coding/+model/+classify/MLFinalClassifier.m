classdef MLFinalClassifier
    
    properties
        training_X, training_y;
    end
    
    methods
        function obj = MLFinalClassifier(X, y)
            obj.training_X = X;
            obj.training_y = y;
        end
        
        function y = predict(obj, X)
            % Define some constants
            training_n = size(obj.training_X, 1);
            predict_n = size(X, 1);
            gamma = 10;
            lamda = 1;
            mu = 1;
            
            % CAT X behind obj.training_X
            longX = [ obj.training_X ; X ];
            longY = [ obj.training_y; zeros(predict_n,1) ];
            long_n = training_n + predict_n;
            
            % Z-Normalization
            longX_mean = repmat( mean(longX), [long_n, 1] );
            longX_std =repmat( std(longX), [long_n, 1] );
            longX = (longX - longX_mean) ./ longX_std;
            
            % Compute K (Gaussian Kernel)
            K = zeros(long_n, long_n);
            for i = 1 : long_n
                for j = i : long_n
                    result = exp(- gamma * norm(longX(i, :) - longX(j, :)) ^ 2);
                    
                    % fill in result
                    K(i, j) = result;
                    K(j, i) = result;
                end
            end
            
            %TTTOOO DDDOOO  S may ~= K
            
            % Compute L (Using Gaussian Similarity)
            S = K;
            D = diag(sum(S, 2));
            L = D - S;
            
            % Compute A (To eliminate unlabeled data)
            A = diag (double( longY~=0 )) ;

            % Use CVX to solve Objective
            c = zeros(long_n, 1);
            cvx_begin
                variable c(long_n)
                minimize((longY - K * c)' * A * (longY - K * c) + mu * (K * c)' * L * (K * c) + lamda * c' * K * c)
            cvx_end
            
            y = K * c;
            y = sign(y);
            y = y(training_n + 1 : long_n);
        end
    end
    
    methods (Static)
        function classifierObj = train(X, y)
            % Construct object
            classifierObj = model.classify.MLFinalClassifier(X, y);
        end
    end
    
end

