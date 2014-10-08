import model.regressor.LinearRegressorLocalWeight

clear %clear workspace

load X.dat
load y.dat

configMap = containers.Map({'tau'}, {100});

myRegressor = LinearRegressorLocalWeight.train(X, y);
value = myRegressor.predict(X, configMap);

%%% plot data %%%
scatter (X, y,'g');
hold on;

plot(X, value, 'ro');
hold off;
