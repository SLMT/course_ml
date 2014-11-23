%%% the dummy classifier predicts the label by randomly assigning a label
%%% based on the ratio of +1/-1 examples in training data

classdef DummyClassifier < handle
   properties
      posRatio; % the ratio of examples with label +1
   end
   
   methods
       function dumClassifierObj = DummyClassifier (pRatio)  % constructor
           dumClassifierObj.posRatio = pRatio;
       end
       function predictedLabel = predict (obj, X)
           predictedLabel = sign(obj.posRatio-rand(length(X),1));
       end
   end
   
   methods (Static)
      function dumClassifierObj = train (X, y)
        dumClassifierObj = model.classify.DummyClassifier(length(X(y==1,:))/length(X));
      end
   end
end