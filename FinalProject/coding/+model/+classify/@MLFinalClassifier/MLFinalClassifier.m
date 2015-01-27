classdef MLFinalClassifier
    
    properties
        training_X;
		training_y;
		training_n;
        
        % Messages
        slient = true;
        
        % Hyperparameters
		gamma_K = 100;
		gamma_S = 100;
		lambda = 0.1;
		miu = 0.1;
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
            
            % Feture selection
			longX = longX( :, 1:3 );
			
            % Compute K (Gaussian Kernel)
            K = model.classify.MLFinalClassifier.getGaussianKernel( longX, obj.gamma_K );
            
            % Compute Similarity Matrix (Similarity)
			S = model.classify.MLFinalClassifier.getGaussianKernel( longX, obj.gamma_S );
			
			% Predicting using cvx and LapRLS
			y = model.classify.MLFinalClassifier.cvxLapSVM( longY, K, S, obj.miu, obj.lambda, true );
			
            y = sign(y);
            y = y(obj.training_n + 1 : long_n);
        end
    end
    
    methods (Static)
        function classifierObj = train(X, y)
            [new_X new_y] = model.classify.MLFinalClassifier.preLabelData(X, y);
            
            %% Call Constructor
            classifierObj = model.classify.MLFinalClassifier(new_X, new_y);
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
        [ y_predicted ] = cvxLapSVM( y, K, S, miu, lambda, silent )
		[ x_LDA_W ] = LDA(Input,Target,Priors)
        [new_X new_y] = preLabelData(X, y)
    end
    
end
