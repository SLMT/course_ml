classdef MLFinalClassifier
    
    properties
        training_X;
		training_y;
		training_n;
        
        % Messages
        slient = true;
        
        % Hyperparameters
		gamma_K = 10;
		gamma_S = 10;
		lambda = 1;
		miu = 1;
        lda_mapping_threshold = 1;
    end
    
    methods
        function obj = MLFinalClassifier(X, y)
            obj.training_X = X;
            obj.training_y = y;
			obj.training_n = size(X,1);
		end
        
        function y = predict(obj, X)
            % Define some constants
            [predict_n, feture_size] = size(X);

            % CAT X behind obj.training_X
            longX = [ obj.training_X ; X ];
            longY = [ obj.training_y; zeros(predict_n,1) ];
            long_n = obj.training_n + predict_n;
            
            % Z-Normalization
            longXmean = mean(longX);
			longXstd = std(longX);
            longX = model.classify.MLFinalClassifier.zNormalize( longX, longXmean, longXstd );
            
            % get labeled data
            labeled_x = longX(longY ~= 0, :);
            labeled_y = longY(longY ~= 0);

            % LDA
            w = LDA(labeled_x, labeled_y);
            for i = 1 : feture_size + 1
                if norm(w(:,i)) <= obj.lda_mapping_threshold
                    w(:,i) = 0;
                end
            end
            longX = [ ones(long_n, 1), longX] * w';
            
            % Compute K (Gaussian Kernel)
            K = model.classify.MLFinalClassifier.getGaussianKernel( longX, obj.gamma_K );
            
            % Compute Similarity Matrix (Similarity)
			S = model.classify.MLFinalClassifier.getGaussianKernel( longX, obj.gamma_S );
			
			% Predicting using cvx and LapRLS
			y = model.classify.MLFinalClassifier.cvxLapRLS( longY, K, S, obj.miu, obj.lambda, obj.slient );
			
            y = sign(y);
            y = y(obj.training_n + 1 : long_n);
        end
    end
    
    methods (Static)
        function classifierObj = train(X, y)
            % Call Constructor
            classifierObj = model.classify.MLFinalClassifier(X, y);
		end
        
        function classifierObj = trainWithParameters(X, y, gamma_K, gamma_S, lambda, miu)
            % Call Constructor
            classifierObj = model.classify.MLFinalClassifier(X, y);
            
            % Set hyperparameters
            classifierObj.gamma_K = gamma_K;
            classifierObj.gamma_S = gamma_S;
            classifierObj.lambda = lambda;
            classifierObj.miu = miu;
		end
		
		[ normalizedX ] = zNormalize( X, Xmean, Xstd )
		[ K ] = getGaussianKernel( X, gamma )
		[ y_predicted ] = cvxLapRLS( y, K, S, miu, lambda, silent )
		[ x_LDA_W ] = LDA(Input,Target,Priors);
    end
    
end
