function [ Y, L, U ] = runMultiLayer( X, W, V )
% RUNMULTILAYER Calculates output and labels of the net
%
%    Inputs:
%              X - Data samples to be classified (matrix)
%              W - Weights of the hidden neurons (matrix)
%              V - Weights of the output neurons (matrix)
%
%    Output:
%              Y - Output for each sample and class (matrix)
%              L - The resulting label of each sample (vector) 
%              U - Activation of hidden neurons (vector)

% Add your own code here
S = X*W; % Calculate the weighted sum of input signals (hidden neuron)

%XTrain = [XTrain, ones(1, size(XTrain, 1))'];

U = tanh(S); % Calculate the activation of the hidden neurons (use hyperbolic tangent)
size(U)

bias = ones(1, size(U, 1))';
U = [U, bias];

size(V)
size(U)
disp("test")
Y = U*V; % Calculate the weighted sum of the hidden neurons
disp("test2")

% Calculate labels
[~, L] = max(Y,[],2);
disp("test3")
end

