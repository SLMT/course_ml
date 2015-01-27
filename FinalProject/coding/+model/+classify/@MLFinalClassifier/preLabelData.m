function [new_X new_y] = preLabelData(X, y)
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
                    tmpClassifier = model.classify.MLFinalClassifier.trainWithParameters(labeled_X, labeled_y, gammaK, gammaS, lambda, miu);
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
    new_X = [labeled_X; unlabeled_X(unlabeled_y ~= 0, :); unlabeled_X(unlabeled_y == 0, :)];
    new_y = [labeled_y; unlabeled_y(unlabeled_y ~= 0, :); unlabeled_y(unlabeled_y == 0, :)];
end

