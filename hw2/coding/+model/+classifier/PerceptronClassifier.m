
classdef PerceptronClassifier < handle
    
    properties
        level; % lifting level
        param; % h is the parameters of hyperplane
    end
   
    methods
        function classifier = PerceptronClassifier (level, param)  % constructor
            classifier.level = level;
            classifier.param = param;
        end
        
        function predictedValue = predict (obj, X)
            num = size(X, 1);
            
            % lifting
            phiX = ones(num, obj.level + 1);
            for i = 1 : obj.level
                phiX(:, i+1) = X.^i;
            end
            
            % predict
            predictedValue = sign(phiX * obj.param);
        end
    end
   
    methods (Static)
        function classifier = train (X, y, liftingLevel, learningRate, phi)
            % initializing variables
            num = size(X, 1);
            
            % lifting
            phiX = ones(num, liftingLevel + 1);
            for i = 1 : liftingLevel
                phiX(:, i+1) = X.^i;
            end
            
            % initialize variables for iteration
            %h = [1; 0.9487; -3.3650]; % update this
            h = ones(liftingLevel + 1, 1);
            updateVector = ones(liftingLevel + 1, 1);
            
            % converage
            while norm(updateVector) > phi
                % predict
                predicted = sign(phiX * h);

                % update h
                updateVector = learningRate * (phiX' * (y - predicted));
                h = h + updateVector;
            end
            
            classifier = model.classifier.PerceptronClassifier(liftingLevel, h);
        end
    end
end