rootFolder = uigetdir;

%load('C:\Users\harri\Documents\MATLAB\NeuralNetwork\NeuralNetwork.mat')
load NeuralNetwork.mat

imds = imageDatastore(fullfile(rootFolder), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
NumberofImages = imds.countEachLabel.Count;

%   Define Struct
field1 = 'Filename';        value1 = [];
field2 = 'LatitudeRef';     value2 = {};
field3 = 'Latitude';        value3 = [];
field4 = 'LongitudeRef';    value4 = {};
field5 = 'Longitude';       value5 = [];
field6 = 'Datetime';        value6 = {};
field7 = 'Classification';  value7 = {};

Classifications = struct(field1, value1, field2, value2, field3, value3, field4, value4, field5, value5, field6, value6, field7, value7);
Classifications(NumberofImages).Latitude = 0;

for i = 1:NumberofImages
   imageInfo = imfinfo(char(imds.Files(i)));
   
   Classifications(i).Filename = imds.Files(i);
   Classifications(i).LatitudeRef = imageInfo.GPSInfo.GPSLatitudeRef;
   Classifications(i).Latitude = imageInfo.GPSInfo.GPSLatitude;
   Classifications(i).LongitudeRef = imageInfo.GPSInfo.GPSLongitudeRef;
   Classifications(i).Longitude = imageInfo.GPSInfo.GPSLongitude;
   Classifications(i).Datetime = imageInfo.DateTime;
   Classifications(i).Classification = classify(netTransfer, readimage(imds, i));
   
end    
Output_File = fullfile( rootFolder, 'Classifications_Output.csv');
Table = struct2table(Classifications);
writetable(Table, Output_File);


