classdef MLFinalClassifier
    
    properties
        training_X;
		training_y;
		training_n;
        training_X_mean;
        training_X_std;
        
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
        function obj = MLFinalClassifier(X, y, X_mean, X_std)
            obj.training_X = X;
            obj.training_y = y;
			obj.training_n = size(X,1);
            obj.training_X_mean = X_mean;
            obj.training_X_std = X_std;
		end
        
        function y = predict(obj, X)
            % Define some constants
            [predict_n, ~] = size(X);
            
            % Feture selection
			X = X( :, 1:3 );
            
            % Z-Normalization
            X = model.classify.MLFinalClassifier.zNormalize( X, obj.training_X_mean, obj.training_X_std );

            % CAT X behind obj.training_X
            longX = [ obj.training_X ; X ];
            longY = [ obj.training_y; zeros(predict_n,1) ];
            long_n = obj.training_n + predict_n;
			
            % Compute K (Gaussian Kernel)
            K = model.classify.MLFinalClassifier.getGaussianKernel( longX, obj.gamma_K );
            
            % Compute Similarity Matrix (Similarity)
			S = model.classify.MLFinalClassifier.getGaussianKernel( longX, obj.gamma_S );
			
			% Predicting using cvx and LapSVM
			y = model.classify.MLFinalClassifier.cvxLapSVM( longY, K, S, obj.miu, obj.lambda, true );
			
            y = sign(y);
            y = y(obj.training_n + 1 : long_n);
        end
    end
    
    methods (Static)
        function classifierObj = train(X, y)
            %% Hyperparameter Sets
            gammaK_set = [100];
            gammaS_set = [100];
            lambda_set =  [0.01 0.1 1];
            miu_set =  [0.01 0.1 1];

            % Feture selection
            X = X( :, 1:3 );
            
            % Z-Normalization
            Xmean = mean(X);
            Xstd = std(X);
            X = model.classify.MLFinalClassifier.zNormalize( X, Xmean, Xstd );

            %% Voting
            % Construct voting matrix
            votes = zeros(size(X, 1), 2);

            % Run all combinations
            fprintf( 'Start to vote. Trying all combination...\n');
            for gk = gammaK_set
                for gs = gammaS_set
                    for l = lambda_set
                        for m = miu_set
                            % Generate labels
                            % Compute K (Gaussian Kernel)
                            K = model.classify.MLFinalClassifier.getGaussianKernel( X, gk );

                            % Compute Similarity Matrix (Similarity)
                            S = model.classify.MLFinalClassifier.getGaussianKernel( X, gs );

                            % Predicting using cvx and LapSVM
                            predict_y = model.classify.MLFinalClassifier.cvxLapSVM( y, K, S, m, l, true );
                            predict_y = sign(predict_y);

                            % Vote
                            votes(predict_y == 1, 1) = votes(predict_y == 1, 1) + 1;
                            votes(predict_y == -1, 2) = votes(predict_y == -1, 2) + 1;
                        end
                    end
                end
            end

            %% Counting votes
            fprintf( 'Voting finished. Counting votes...\n');
            total_vote = votes(1, 1) + votes(1, 2);
            good_vote = total_vote * 0.8;

            % Use votes to decide labels
            vote_y = zeros(size(y, 1), 1);
            vote_y(votes(:, 1) > good_vote) = 1;
            vote_y(votes(:, 2) > good_vote) = -1;

            % Calculate labeled data's accurancy
            correct = size(y(y ~= 0 & y == vote_y), 1);
            total = size(y(y ~= 0), 1);
            accurancy = correct / total;

            % Show results
            fprintf( 'Accurancy in labeled data: %.2f%%\n', accurancy * 100);
            fprintf( 'Label %d data as 1, %d data as -1, %d data are still unlabeled.\n', size(y(vote_y == 1 & y == 0), 1), size(y(vote_y == -1 & y == 0), 1), size(y(vote_y == 0 & y == 0), 1));

            %% Make unlabeled data to labeled data
            new_y = y;
            new_y(y == 0) = vote_y(y == 0);
            
            %% Call Constructor
            classifierObj = model.classify.MLFinalClassifier(X, new_y, Xmean, Xstd);
        end
        
        function classifierObj = trainWithParameters(X, y, gamma_K, gamma_S, lambda, miu)
            % Feture selection
            X = X( :, 1:3 );
            
            % Z-Normalization
            Xmean = mean(X);
            Xstd = std(X);
            X = model.classify.MLFinalClassifier.zNormalize( X, Xmean, Xstd );
            
            % Call Constructor
            classifierObj = model.classify.MLFinalClassifier(X, y, Xmean, Xstd);
            
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
