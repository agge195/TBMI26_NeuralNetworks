function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);
cM = confusionmat(LTrue, LPred);
cM

end

