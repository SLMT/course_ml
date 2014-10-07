import model.regressor.LinearRegressor

clear %clear workspace

X = random('norm', 5, 1, [100000000, 1]);
y = ones(100000000, 1);

tic
myRegressor = LinearRegressor.train(X,y);
toc