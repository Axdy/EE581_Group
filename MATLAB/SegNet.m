% SegNet.m
% Uses a pretrained SegNet dataset to classify images of road scenes.
clear;
%% Download pretrained SegNet classifier - Trained with CamVid dataset

% pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/segnetVGG16CamVid.mat';
% pretrainedFolder = fullfile(tempdir,'pretrainedSegNet');
% pretrainedSegNet = fullfile(pretrainedFolder,'segnetVGG16CamVid.mat'); 
% if ~exist(pretrainedFolder,'dir')
%     mkdir(pretrainedFolder);
%     disp('Downloading pretrained SegNet (107 MB)...');
%     websave(pretrainedSegNet,pretrainedURL);
% end
%% Load SegNet CNN - Trained with CamVid dataset
addpath('pretrainedSegNet/');
load('pretrainedSegNet/segnetVGG16CamVid.mat');

%% Set up classes

classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Cyclist"
    ];

%% setup colour map for displaying results

cmap = [
    128 128 128   % Sky
    128 0 0       % Building
    192 192 192   % Pole
    128 64 128    % Road
    60 40 222     % Pavement
    128 128 0     % Tree
    192 128 128   % SignSymbol
    64 64 128     % Fence
    64 0 128      % Car
    64 64 0       % Pedestrian
    0 128 192     % Bicyclist
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;

%% read in test image - Run classifier
I = imread('glasgow.jpg');
I = imresize(I, [480, 640]);
J = imread('kelvinway.png');
J = imresize(J, [480, 640]);
K = imread('tic.png');
K = imresize(L, [480, 640]);
L = imread('snow.jpg');
L = imresize(L, [480, 640]);
C = semanticseg(I, net);    %Run Classifier
D = semanticseg(J, net);    %Run Classifier
E = semanticseg(K, net);    %Run Classifier
F = semanticseg(L, net);    %Run Classifier

%% Display segmented image - West George Street
B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency', 0.1);
figure;
imshow(I);
title('original');
figure;
imshow(B);
pixelLabelColorbar(cmap, classes);
title('segmented');

%% Display segmented image two - Kelvin Way
A = labeloverlay(J, D, 'Colormap', cmap, 'Transparency', 0.1);
figure;
imshow(J);
title('original');
figure;
imshow(A);
pixelLabelColorbar(cmap, classes);
title('segmented');

%% Display segmented image three - TIC
Z = labeloverlay(K, E, 'Colormap', cmap, 'Transparency', 0.1);
figure;
imshow(K);
title('original');
figure;
imshow(Z);
pixelLabelColorbar(cmap, classes);
title('segmented');

%% Display segmented image four - Snowy Glasgow
Y = labeloverlay(L, F, 'Colormap', cmap, 'Transparency', 0.1);
figure;
imshow(L);
title('original');
figure;
imshow(Y);
pixelLabelColorbar(cmap, classes);
title('segmented');