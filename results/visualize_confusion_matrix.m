clear; close all; clc; 
confmat = [ 822,  12,  23,  12,  12,   5,   5,   9,  66,  34;
   4, 935,   0,   2,   1,   1,   2,   0,   5,  50;
  33,   0, 784,  27,  61,  47,  22,  12,   9,   5;
   7,   0,  28, 719,  42, 128,  30,  19,  15,  12;
   2,   1,  20,  29, 892,  13,  12,  23,   6,   2;
   4,   2,  11,  78,  27, 839,   9,  19,   9,   2;
   2,   2,  20,  42,  22,   9, 895,   5,   1,   2;
   4,   0,   8,  22,  32,  24,   2, 899,   3,   6;
  20,  13,   4,   2,   2,   0,   3,   2, 927,  27;
   8,  34,   2,   3,   0,   0,   1,   0,   9, 943;
];

% plotting
plotConfusionMatrix(confmat, { 'Airplane', 'Auto', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Horse', 'Ship', 'Truck',});


%%
function plotConfusionMatrix(varargin)
%PLOTCONFMAT plots the confusion matrix with colorscale, absolute numbers
%and precision normalized percentages
%
%   usage: 
%   PLOTCONFMAT(confmat) plots the confmat with integers 1 to n as class labels
%PLOTCONFMAT(confmat, labels) plots the confmat with the specified labels
%
%   Arguments
%   confmat:			a square confusion matrix
%labels (optional):  vector of class labels

% number of arguments
switch (nargin)
	case 0
		confmat = 1;
		labels = {'1'};
	case 1
		confmat = varargin{1};
		labels = 1:size(confmat, 1);
	otherwise
		confmat = varargin{1};
		labels = varargin{2};
end

confmat(isnan(confmat))=0; % in case there are NaN elements
numlabels = size(confmat, 1); % number of labels

% calculate the percentage accuracies
confpercent = 100*confmat./sum(confmat, 2);

% plotting the colors
imagesc(confpercent);
title(sprintf('Accuracy: %.2f%%', 100*trace(confmat)/sum(confmat(:))));
ylabel('Predicted Category','FontSize',10); xlabel('Target Category','FontSize',10);

% set the colormap
colormap(flipud(hot));
colorbar;

% Create strings from the matrix values and remove spaces
textStrings = num2str([confpercent(:), confmat(:)], '%.1f%%\n%d');
textStrings = strtrim(cellstr(textStrings));

% Create x and y coordinates for the strings and plot them
[x,y] = meshgrid(1:numlabels);
hStrings = text(x(:),y(:),textStrings(:), ...
'HorizontalAlignment','center');

% Get the middle value of the color range
midValue = mean(get(gca,'CLim'))

% Choose white or black for the text color of the strings so
% they can be easily seen over the background color
textColors = repmat(confpercent(:) > midValue,1,3);
set(hStrings,{'Color'},num2cell(textColors,2));

% Setting the axis labels
truesize([550 650]);
set(gca,'XTick',1:numlabels,...
		'XTickLabel',labels,...
		'YTick',1:numlabels,...
		'YTickLabel',labels,...
		'TickLength',[0 0]);
end
