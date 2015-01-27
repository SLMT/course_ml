import model.classify.MLFinalClassifier

clear; close; clc;

%% import data set
load data/X.mat
load data/y.mat
load data/Xtest.mat
load data/ytest.mat

%% Hyperparameters
gammaK_set = [0.01 0.1 1 10 100];
gammaS_set = gammaK_set;
lambda_set = gammaK_set;
miu_set = gammaK_set;

%% Find semi-labeled data
for gammaK = gammaK_set
    for gammaS = gammaS_set
        for lambda = lambda_set
            for miu = miu_set
                
            end
        end
    end
end

%% Training
myClassifier = MLFinalClassifier.train(X, y);

%% Make prediction
label = myClassifier.predict(Xtest);

%% show error rate
accuracy = 1 - size(label(label ~= ytest), 1) / size(label, 1);
fprintf( 'Accuracy : %.2f%%\n', accuracy*100 );
