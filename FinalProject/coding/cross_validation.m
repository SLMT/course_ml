import model.classify.MLFinalClassifier

clear; close; clc;

%% import data set
load data/X.mat
load data/y.mat

%% Hyperparameter Sets
gammaK_set = [100];
gammaS_set = [100];
lambda_set =  [0.1];
miu_set =  [0.001];

%% Find semi-labeled data
% Split data set into labeled and unlabeled
labeled_X = X(y ~= 0, :);
labeled_y = y(y ~= 0);
unlabeled_X = X(y == 0, :);

% Assume all unlabeled data are semi-labeled data
is_semi_labeled = ones(size(unlabeled_X, 1), 1);

% Run all combinations
fprintf( 'Start to find semi-labeled data\n');
lastY = 0;
for gammaK = gammaK_set
	for gammaS = gammaS_set
        for lambda = lambda_set
            for miu = miu_set
                % Generate labels
                tmpClassifier = MLFinalClassifier.trainWithParameters(labeled_X, labeled_y, gammaK, gammaS, lambda, miu);
                predict_y = tmpClassifier.predict(unlabeled_X);
                
                % Find semi-labeled data
                if (lastY ~= 0)
                    is_semi_labeled(is_semi_labeled == 1 & lastY ~= predict_y) = 0;
                end
                lastY = predict_y;
            end
        end
	end
end

%% Make semi-labeled data to labeled data
fprintf( 'Find out %d semi-labeled data\n', size(is_semi_labeled(is_semi_labeled == 1), 1));
labeled_X = [labeled_X; unlabeled_X(is_semi_labeled == 1, :)];
labeled_y = [labeled_y; lastY(is_semi_labeled == 1)];
labeled_n = size(labeled_X, 1);

unlabeled_X = unlabeled_X(is_semi_labeled == 0, :);
unlabeled_n = size(unlabeled_X, 1);
unlabeled_y = zeros(unlabeled_n, 1);

%% Find out best combination
% Determine folds
foldsNum= 5;
labeled_folds_index = mod(randsample(labeled_n, labeled_n), foldsNum);
unlabeled_folds_index = mod(randsample(unlabeled_n, unlabeled_n), foldsNum);

% Run all combinations
for gammaK = gammaK_set
	for gammaS = gammaS_set
        for lambda = lambda_set
            for miu = miu_set
                
                % Cross Validation
                accuracy = 0;
                for k = 0 : foldsNum - 1
                    % Get data set
                    training_X = [labeled_X(labeled_folds_index ~= k, :); unlabeled_X(unlabeled_folds_index ~= k, :)];
                    training_y = [labeled_y(labeled_folds_index ~= k, :); unlabeled_y(unlabeled_folds_index ~= k, :)];
                    testing_X = [labeled_X(labeled_folds_index == k, :); unlabeled_X(unlabeled_folds_index == k, :)];
                    testing_y = [labeled_y(labeled_folds_index == k, :); unlabeled_y(unlabeled_folds_index == k, :)];
                    
                    % Training & Prediction
                    tmpClassifier = MLFinalClassifier.trainWithParameters(training_X, training_y, gammaK, gammaS, lambda, miu);
                    predict_y = tmpClassifier.predict(testing_X);
                    
                    % Calculate accuracy
                    currect = size(testing_y(testing_y ~= 0 & testing_y == predict_y), 1);
                    total = size(testing_y(testing_y ~= 0), 1);
                    accuracy = accuracy + currect / total;
                end
                
                
                % Show accuracy
                accuracy = accuracy / foldsNum;
                fprintf( 'Accuracy when gk=%.3f, gs=%.3f, l=%.3f, m=%.3f : %.2f%%\n', gammaK, gammaS, lambda, miu, accuracy*100 );
            end
        end
	end
end
