%%% the dummy regressor predicts the value by mean value of y

classdef DummyRegressor < handle
   properties
      meanValue; % the mean value we predict
   end
   
   methods
       function dummyRegressorObj = DummyRegressor (mean)  % constructor
           dummyRegressorObj.meanValue = mean;
       end
       function predictedValue = predict (obj, X)
           for i = 1:length(X)
		       predictedValue(i) = obj.meanValue;
		   end
       end
   end
   
   methods (Static)
      function dummyRegressorObj = train (X, y)
        dummyRegressorObj = model.regressor.DummyRegressor(mean(y));
      end
   end
end