import model.classify.DummyClassifier

clear %clear workspace

load data/X.dat
load data/y.dat

%--- TODO: please import the dataset here ---%

%--- TODO: modify the DummyClassifier to your SoftMarginLinearClassifier ---%
%---       please follow the specs strickly                              ---%
myClassifier = DummyClassifier.train(X,y);
label = myClassifier.predict(X);

%%% plot data %%%
scatter (X(y==1,1),X(y==1,2),'g');
hold on;
scatter (X(y==-1,1),X(y==-1,2),'b');

%--- TODO: plot the decision boundary yourself ---%


hold off;
