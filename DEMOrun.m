clear;close all;
%% add paths
addpath(genpath('utils'));
addpath(genpath('matconvnet-1.0-beta25\matlab')) ;

%% set params
dataDir = '.\ExampleImage\';
pz = 3; 

rng(791);
colors = distinguishable_colors(23);
colors(1:4 ,:)=[];
colors([3 5 7 8] ,:)=[];

%% load images and target location
pairNum = 230;
swap = false;
displayGT = true;
[I,Iref,T,rectT,rectGT] = utils.loadImageAndTemplate(pairNum,dataDir, swap); % more examples can be found in the BBS datasetes

fprintf('T size %dX%d\n', size(T,1),size(T,2));
fprintf('Target image size %dX%d\n', size(I,1),size(I,2));

%% adjust image and template size so they are divisible by the patch size 'pz'
szI = size(I);
disArr = {};
Rects = {};
Names = {};
ind=1;
%% run (ucomment the different methods to run them)

%-------------------------------------------------------------
tic;
[heatmap, rectWSDIS]   = computeWSDIS(I, T, pz); %core function
runtime(ind)=toc;
disArr{ind}=heatmap;
Rects{ind}=rectWSDIS;
Names{ind} = 'WSDIS RGB';
ind=ind+1;
%-------------------------------------------------------------
if ~exist('net','var')
    [ net, gpuN ] = loadNet();    % loading imagenet-vgg-verydeep-19.mat
end
tic;
[heatmap, rectWSDIS]   = computeWSDIS_deep(I,T, net, gpuN, 'L2',0); %core function
runtime(ind)=toc;
disArr{ind}=heatmap;
Rects{ind}=rectWSDIS;
Names{ind} = 'WSDIS deep L2';
ind=ind+1;
%-------------------------------------------------------------

total = length(Rects);
colors = colors(1:total+2,:);

%% compute overlap with ground-truth
Overlaps = {};
for i=1:length(Names)
    Overlaps{i} = rectOverlap(rectCorners(rectGT), rectCorners(Rects{i}));
end

%% plot results
f1 = figure;
rectWidth = 1;
imshow(Iref);hold on;
r = rectangle('position',rectT,'linewidth',rectWidth,'edgecolor',[0 1 0]);
plot(nan,nan,'s','markeredgecolor',get(r,'edgecolor'),'markerfacecolor',get(r,'edgecolor'),'linewidth',3);hold off;

f2 = figure;
imshow(I);hold on;
for j=1:total
    rct(j) = rectangle('position',Rects{j},'linewidth',rectWidth,'edgecolor',colors(j,:));
    plt(j) = plot(nan,nan,'s','markeredgecolor',get(rct(j),'edgecolor'),'markerfacecolor',get(rct(j),'edgecolor'),'linewidth',3);
    NamesWithOL{j} = [Names{j}, sprintf(' Ol=%.2f',Overlaps{j})];
end

j = total+1;
if displayGT
    rct(j) = rectangle('position',rectGT,'linewidth',rectWidth,'edgecolor',[0,1,0]);
    plt(j) = plot(nan,nan,'s','markeredgecolor',get(rct(j),'edgecolor'),'markerfacecolor',get(rct(j),'edgecolor'),'linewidth',3);
    Names{j}='GroundTruth';
    NamesWithOL{j}='GroundTruth';
end
legend(plt,NamesWithOL,'location','southeast');set(gca,'fontsize',6);



clear mex


