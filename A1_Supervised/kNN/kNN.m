function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);
    
   euclideanDistances = pdist2(X, XTrain, 'euclidean'); 
    
   [dist, idx] = sort(euclideanDistances, 2); 

   neighbors = LTrain(idx(:, 2:k+1)); % get k-nearest from index
   
   [LPred, F, C] = mode(neighbors, 2);  
   
   for i = 1:size(C)
    if size(C{i}, 1) > 1
       if LPred(i) ~= neighbors(i)
         LPred(i) = neighbors(i);
       end
    end
   end
   
   
end