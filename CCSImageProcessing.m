% Get GPU device information
deviceInfo = gpuDevice;

% Check the GPU compute capability
computeCapability = str2double(deviceInfo.ComputeCapability);
assert(computeCapability >= 3.0, ...
    'This example requires a GPU device with compute capability 3.0 or higher.')

outputFolder = fullfile('C:\Users\harri\Documents\MATLAB\NeuralNetwork', 'caltech101'); % define output folder
rootFolder = fullfile('C:\Users\harri\Documents\MATLAB\NeuralNetwork', '101_ObjectCategories');
% categories = {'airplanes', 'ferry', 'laptop'};
categories = {'wet', 'dry'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

%   Creates the same number of images for each Label by selecting
%   minsetcount randomized images 
imds = splitEachLabel(imds, minSetCount, 'randomize');

% countEachLabel(imds)

% Find the first instance of an image for each category
% airplanes = find(imds.Labels == 'airplanes', 1);
% ferry = find(imds.Labels == 'ferry', 1);
% laptop = find(imds.Labels == 'laptop', 1);
dry = find(imds.Labels == 'dry', 1);
wet = find(imds.Labels == 'wet', 1);

% figure
% subplot(1,3,1);
% imshow(readimage(imds,airplanes))
% subplot(1,3,2);
% imshow(readimage(imds,ferry))
% subplot(1,3,3);
% imshow(readimage(imds,laptop))



%   Load pre-trained Neural Net AlexNet
net = alexnet();

%   Pre-process images to RGB 227x227 specifically for Alexnet
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%   Randomize data into training and validation data (50/50)
[trainingSet, testSet] = splitEachLabel(imds, 0, 'randomize');
 % Visual Representation of second convolution Layer

% % Get the network weights for the second convolutional layer
% w1 = netTransfer.Layers(3).Weights;
% 
% % Scale and resize the weights for visualization
% w1 = mat2gray(w1);
% w1 = imresize(w1,5);
% 
% % Display a montage of network weights. There are 96 individual sets of
% % weights in the first layer.
% figure
% montage(w1)
% title('First convolutional layer weights')
%% % 
featureLayer = 'fc7';
trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Extract test features using the CNN
testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy
mean(diag(confMat))