
classdef PerceptronClassifier < handle
   properties
      
   end
   
   methods
       function classifier = PerceptronClassifier ()  % constructor
           
       end
       function predictedValue = predict (obj, X)
           
       end
   end
   
   methods (Static)
      function classifier = train (X, y, liftingLevel)
      	classifier = model.classifier.PerceptronClassifier();
      end
   end
end