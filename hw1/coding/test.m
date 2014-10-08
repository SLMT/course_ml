import model.regressor.LinearRegressor

clear %clear workspace

%--- TODO: please import the dataset here ---%
load X.dat
load y.dat

X = 

%--- TODO: modify the DummyRegressor to your LinearRegressor & LinearRegressorLocalWeight ---%
%---       please follow the specs strickly                              ---%
myRegressor1 = LinearRegressor.train(X,y);
myRegressor2 = LinearRegressor.gradientDescentTrain(X,y);

value1 = myRegressor1.predict(X);
value2 = myRegressor2.predict(X);

emp1 = LinearRegressor.calculateEMP(myRegressor1.w, X, y)
emp2 = LinearRegressor.calculateEMP(myRegressor2.w, X, y)

%%% plot data %%%
scatter (X, y,'g');
hold on;

%--- TODO: plot the regressor you train ---%
plot(X, value1, 'r');
plot(X, value2, 'r');
hold off;
