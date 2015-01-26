classdef MLFinalClassifier
    
    properties
        training_X;
		training_LDA_W;
		training_y;
		training_n;
		training_X_Mean;
		training_X_Std;
		gamma_K = 10;
		gamma_S = 10;
		lambda = 1;
		miu = 1;
    end
    
    methods
        function obj = MLFinalClassifier(X, y, Xmean, Xstd)
            obj.training_X = X;
            obj.training_y = y;
			obj.training_n = size(X,1);
			obj.training_X_Mean = Xmean;
			obj.training_X_Std = Xstd;
		end
        
        function y = predict(obj, X)
            % Define some constants
            predict_n = size(X, 1);
            
			% Normalize testing data ( Z-Normalization )
			X = model.classify.MLFinalClassifier.zNormalize( X, obj.training_X_Mean, obj.training_X_Std );
			
            % CAT X behind obj.training_X
            longX = [ obj.training_X ; X ];
            longY = [ obj.training_y; zeros(predict_n,1) ];
            long_n = obj.training_n + predict_n;
            
            % Compute K (Gaussian Kernel)
            K = model.classify.MLFinalClassifier.getGaussianKernel( longX, obj.gamma_K );
            
            % Compute Similarity Matrix (Similarity)
			S = model.classify.MLFinalClassifier.getGaussianKernel( longX, obj.gamma_S );
			
			% Predicting using cvx and LapRLS
			y = model.classify.MLFinalClassifier.cvxLapRLS( longY, K, S, obj.miu, obj.lambda, false );
			
            y = sign(y);
            y = y(obj.training_n + 1 : long_n);
        end
    end
    
    methods (Static)
        function classifierObj = train(X, y)
            % Call Constructor
			Xmean = mean(X);
			Xstd = std(X);
			normalizedX = model.classify.MLFinalClassifier.zNormalize( X, Xmean, Xstd );
            classifierObj = model.classify.MLFinalClassifier(normalizedX, y, Xmean, Xstd);
		end
		
		[ normalizedX ] = zNormalize( X, Xmean, Xstd )
		[ K ] = getGaussianKernel( X, gamma )
		[ y_predicted ] = cvxLapRLS( y, K, S, miu, lambda, silent )
		[ x_LDA_W ] = LDA(Input,Target,Priors);
    end
    
end
