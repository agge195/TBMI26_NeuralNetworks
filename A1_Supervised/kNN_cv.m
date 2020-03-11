%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

%% Select a subset of the training samples

numBins = 2;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);
% Add your own code to setup data for training and test here
XTrain = XBins{1};
LTrain = LBins{1};
XTest  = XBins{2};
LTest  = LBins{2};

%% Use kNN to classify data
%  Note: you have to modify the kNN() function yourself.

% Set the number of neighbors
k = 40;
res = zeros(k, 2);

for i = 1:k
    acc = 0;
    
    for j = 1:numBins
        
        if j ==1
            kNN_res = kNN(XTest, i, XTrain, LTrain);
        elseif j == 2
            kNN_res = kNN(XTrain, i, XTrain, LTrain);
        end
        
        cM = calcConfusionMatrix(kNN_res, LTest);
        acc = calcAccuracy(cM); 
        
    end
    
    result(i, 2) = acc;
    result
end


plot(result(:, 2))
xlabel("k-value")
ylabel("accuracy")
