function [Wout,Vout,ErrTrain,ErrTest] = trainMultiLayer(XTrain,DTrain,XTest,DTest,W0,V0,numIterations,learningRate)
% TRAINMULTILAYER Trains the multi-layer network (Learning)
%    Inputs:
%                X* - Training/test samples (matrix)
%                D* - Training/test desired output of net (matrix)
%                V0 - Initial weights of the output neurons (matrix)
%                W0 - Initial weights of the hidden neurons (matrix)
%                numIterations - Number of learning steps (scalar)
%                learningRate  - The learning rate (scalar)
%
%    Output:
%                Wout - Weights after training (matrix)
%                Vout - Weights after training (matrix)
%                ErrTrain - The training error for each iteration (vector)
%                ErrTest  - The test error for each iteration (vector)

% Initialize variables
ErrTrain = nan(numIterations+1, 1);
ErrTest  = nan(numIterations+1, 1);
NTrain   = size(XTrain, 1);
NTest    = size(XTest , 1);
NClasses = size(DTrain, 2) - 1;
Wout = W0;
Vout = V0;

% Calculate initial error
YTrain = runMultiLayer(XTrain, W0, V0);
YTest  = runMultiLayer(XTest , W0, V0);

size(YTrain)
size(DTrain)
%YTrain ska vara 1000x2
ErrTrain(1) = sum(sum((YTrain - DTrain).^2)) / (NTrain * NClasses);
ErrTest(1)  = sum(sum((YTest  - DTest ).^2)) / (NTest  * NClasses);

for n = 1:numIterations
    % Add your own code here
    S = XTrain*Wout;
    bias =  ones(1, size(tanh(S), 1))';
    U = [tanh(S), bias];
    %XTrain = [XTrain, ones(1, size(XTrain, 1))'];
    size(U)
    Y = U*Vout;
    size(Y)
    
    tanh_d = 1 - tanh(S).^2;
    Vout_noBias = Vout(2:length(Vout), :);
    
    size(Vout_noBias)
    
    grad_v = (2/numIterations)*U'*(Y - DTrain); % Gradient for the output, vet ej om korrekt
    
    grad_w = (2/numIterations)*(XTrain'*(((Y - DTrain)*Vout_noBias').*tanh_d))
    
    
  %  grad_w = (2/numTraining)*(((red_Vout'*(Y - Dtraining)).*(du))*Xtraining');
    
    % Take a learning step
    Vout = Vout - learningRate * grad_v;
    Wout = Wout - learningRate * grad_w;
    
    % Evaluate errors
    YTrain = runMultiLayer(XTrain, Wout, Vout);
    YTest  = runMultiLayer(XTest , Wout, Vout);
    ErrTrain(1+n) = sum(sum((YTrain - DTrain).^2)) / (NTrain * NClasses);
    ErrTest(1+n)  = sum(sum((YTest  - DTest ).^2)) / (NTest  * NClasses);
end

end