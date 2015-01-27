import model.classify.MLFinalClassifier

clear; close; clc;

%% import data set
load data/X.mat
load data/y.mat

%% Hyperparameter Sets
gammaK_set = [100];
gammaS_set = [100];
lambda_set =  [0.01 0.1 1];
miu_set =  [0.01 0.1 1];

%% Find semi-labeled data
% Split data set into labeled and unlabeled
labeled_X = X(y ~= 0, :);
labeled_y = y(y ~= 0);
unlabeled_X = X(y == 0, :);

% Construct voting matrix
votes = zeros(size(unlabeled_X, 1), 2);

% Run all combinations
fprintf( 'Start to vote. Trying all combination...\n');
lastY = 0;
for gammaK = gammaK_set
	for gammaS = gammaS_set
        for lambda = lambda_set
            for miu = miu_set
                % Generate labels
                tmpClassifier = MLFinalClassifier.trainWithParameters(labeled_X, labeled_y, gammaK, gammaS, lambda, miu);
                predict_y = tmpClassifier.predict(unlabeled_X);
                
                % Vote
                votes(predict_y == 1, 1) = votes(predict_y == 1, 1) + 1;
                votes(predict_y == -1, 2) = votes(predict_y == -1, 2) + 1;
            end
        end
	end
end

%% Decide unlabeled data's labels
fprintf( 'Voting finished. Counting votes...\n');
total_vote = votes(1, 1) + votes(1, 2);
good_vote = total_vote * 0.8;

unlabeled_n = size(unlabeled_X, 1);
unlabeled_y = zeros(unlabeled_n, 1);
unlabeled_y(votes(:, 1) > good_vote) = 1;
unlabeled_y(votes(:, 2) > good_vote) = -1;

% Show results
fprintf( 'Label %d data as 1, %d data as -1, %d data are still unlabeled.\n', size(unlabeled_y(unlabeled_y == 1), 1), size(unlabeled_y(unlabeled_y == -1), 1), size(unlabeled_y(unlabeled_y == 0), 1));

%% Make unlabeled data to labeled data
labeled_X = [labeled_X; unlabeled_X(unlabeled_y ~= 0, :)];
labeled_y = [labeled_y; unlabeled_y(unlabeled_y ~= 0, :)];
labeled_n = size(labeled_X, 1);

unlabeled_X = unlabeled_X(unlabeled_y == 0, :);
unlabeled_n = size(unlabeled_X, 1);
unlabeled_y = zeros(unlabeled_n, 1);

%% Find out best combination
% Determine folds
foldsNum= 5;
labeled_folds_index = mod(randsample(labeled_n, labeled_n), foldsNum);
unlabeled_folds_index = mod(randsample(unlabeled_n, unlabeled_n), foldsNum);

% Run all combinations
fprintf( 'Start cross validation\n');
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
