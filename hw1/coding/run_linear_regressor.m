import model.regressor.LinearRegressor

clear %clear workspace

load X.dat
load y.dat

myRegressor = LinearRegressor.train(X,y);
value = myRegressor.predict(X);

empericalError = LinearRegressor.calculateEMP(myRegressor.w, X, y)

%%% plot data %%%
scatter (X, y,'g');
hold on;

plot(X, value, 'r');
hold off;
