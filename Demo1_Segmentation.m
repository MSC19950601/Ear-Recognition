%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Charles Chen - SID: 810797715
% CSCI5722 - Computer Vision
% Final Project - Ear Segmentation Demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Based on work of Milad Lankarany and Alireza Ahmadyfard:
% "Ear Segmentation Using Topographic Labels"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
disp('Begin Ear Detection and Segmentation');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup Workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output_dir = 'outputs/segmentation_demo/';
image_dir = 'images/';

tic
% target = imread('images/profile_f1.png');
% target = imread('images/profile_m1.png');
target = imread('images/profile_m2.png');
% target = imread('images/profile_m3.png');

figure('Name','Input Image');imshow(target);

target_bw = rgb2gray(target);
imwrite(target_bw,strcat(output_dir,'Seg_00_initial.png'),'png');

% Apply a light Gaussian Blur
gauss = fspecial('gaussian',6,5);
blurred = imfilter(target_bw,gauss);
f = histeq(double(blurred)/255.)*255;
% figure('Name','Input Image');imshow(f/255.);
clear gauss;
imwrite(f/255.,strcat(output_dir,'Seg_00_preprocessed.png'),'png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Topographic Labeling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set up Derivatives
ddx2 = [1 -2 1];
ddy2 = [1 -2 1]';
ddxy = [-1/4 0 1/4; 0 0 0; 1/4 0 -1/4];
ddx = [-1/2 0 1/2];
ddy = [1/2 0 -1/2]';

dHdx = conv2(f,ddx,'same');
dHdy = conv2(f,ddy,'same');

dHdx2 = conv2(f,ddx2,'same');
dHdy2 = conv2(f,ddy2,'same');
dHdxy = conv2(f,ddxy,'same');

% Calculate Hessian
H = zeros(2,2,size(f,1),size(f,2));
H(1,1,:,:) = dHdx2;
H(1,2,:,:) = dHdxy;
H(2,1,:,:) = dHdxy;
H(2,2,:,:) = dHdy2;

% Calculate Gradient
grad_f = zeros(1,2,size(f,1),size(f,2));
grad_f(1,1,:,:) = dHdx;
grad_f(1,2,:,:) = dHdy;

% Assign Labels to Image based on Gradient and Hessian Values
labels1 = zeros(size(f,1),size(f,2));
labels2 = zeros(size(f,1),size(f,2));
labels3 = zeros(size(f,1),size(f,2));

for i=1:size(f,1)
    for j=1:size(f,2)
        % Perform Single Value Decomposition on Hessian
        [u s v] = svd(H(:,:,i,j));
        if ((norm(grad_f(:,:,i,j)) == 0)&&(s(1,1) > 0)&&(s(2,2)==0))||...
                ((norm(grad_f(:,:,i,j)) ~= 0)&&(s(1,1) > 0)&&( grad_f(:,:,i,j)*u(:,1)==0))||...
                ((norm(grad_f(:,:,i,j)) == 0)&&(s(2,2) > 0)&&( grad_f(:,:,i,j)*u(:,2)==0))
            labels1(i,j) = 1;
        elseif ((norm(grad_f(:,:,i,j)) ~= 0)&&(s(1,1) > 0)&&(s(2,2) > 0))||...
                ((norm(grad_f(:,:,i,j)) ~= 0)&&(s(1,1) > 0)&&(s(2,2) == 0))
            labels2(i,j) = 1;
        elseif ((norm(grad_f(:,:,i,j)) ~= 0)&&(s(1,1) > 0)&&(s(2,2) < 0))
            labels3(i,j) = 1;
        end
    end
end

% Threshold the labeled points
post_thresh1 = im2bw(f.*labels1/255, 150./255.);
post_thresh2 = im2bw(f.*labels2/255, 150./255.);
post_thresh3 = im2bw(f.*labels3/255, 150./255.);
% figure;imshow(post_thresh1);
% figure;imshow(post_thresh2);
% figure;imshow(post_thresh3);
imwrite(post_thresh1,strcat(output_dir,'Seg_01_label_1.png'),'png');
imwrite(post_thresh2,strcat(output_dir,'Seg_01_label_2.png'),'png');
imwrite(post_thresh3,strcat(output_dir,'Seg_01_label_3.png'),'png');

% Combine the three labeled masks and reduce to a binary image
merged = post_thresh1 + post_thresh2 + post_thresh3;
merged = merged ~= 0;
figure('Name','Merged Label Groups');imshow(merged);
imwrite(merged,strcat(output_dir,'Seg_02_merged.png'),'png');

% Calculate Erosion and Dilation Difference to Fine Tune Mask
se = [1 0 0; 1 0 0; 1 1 1];
eroded = imerode(f/255,se);
dilated = imdilate(f/255,se);

difference = max(dilated-eroded,0);
figure('Name','Erosion and Dilation Difference');imshow(difference);
imwrite(difference,strcat(output_dir,'Seg_03_DE_difference.png'),'png');

% Combine Erosion/Dilation Difference with Previous Mask and Threshold
improved = im2bw(difference.*merged, 20./255.);
figure('Name','Combined Mask');imshow(improved);
imwrite(improved,strcat(output_dir,'Seg_04_improved_mask.png'),'png');

% Blur and Merge Nearby White Areas in Mask
gauss = fspecial('gaussian',2,2);
expanded = (imfilter(improved,gauss)) ~= 0;
for i=1:10
expanded = (imfilter(expanded,gauss)) ~= 0;
end
% Mask out a 10px margin around the image edge
edge_mask = zeros(size(expanded,1),size(expanded,2));
edge_mask(10:end-10,10:end-10) = 1;
focus_mask = expanded .* edge_mask;
clear expanded edge_mask;
figure('Name','Expanded Mask');imshow(focus_mask);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identify Connected Regions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Label Distinct Regions and Merge Nearby Regions
[L, region_count] = bwlabel(focus_mask, 8);
while (region_count > 65)
    focus_mask = (imfilter(focus_mask,gauss)) ~= 0;
	[L, region_count] = bwlabel(focus_mask, 8);
end

imwrite(focus_mask,strcat(output_dir,'Seg_05_expanded_mask.png'),'png');

% Display Distinct Regions
labled_figure=figure('Name','Marked Distinct Regions');imshow(focus_mask)
hold on;
for r=1:region_count
[y, x] = find(L==r);
scatter(x,y,'.')
axis([1 size(f,2) 1 size(f,1)]);
set(gca,'YDir','reverse');
end
hold off;
saveas(labled_figure,strcat(output_dir,'Seg_06_labeled_mask.png')) 

% My Contribution: Improve Segmentated Selection for
% Images with other distracting background features
std_mean_list = zeros(region_count,1);
cropped_list = cell(region_count,1);
xdim = size(target,2);
ydim = size(target,1);
for r=1:region_count
    % For each distinct region, crop image to that region
    [y, x] = find(L==r);
    xmin = min(x);
    ymin = min(y);
    width = max(x) - xmin;
    height = max(y) - ymin;
    
    % Add 5 pixel margin around actual region taken
    margin = 5;
    xmin2 = max(0,xmin-margin);
    ymin2 = max(0,ymin-margin);
    width2 = min(width + 2*margin, xdim - xmin);
    height2 = min(height + 2*margin, ydim - ymin);
    
    % Ignore Excessively Small Regions
    if (width >= 40)&&(height >= 40)
        cropped = imcrop(f/255, [xmin ymin width height]);
        % Save Cropped Region
        cropped_list{r,1} = imcrop(target_bw, [xmin2 ymin2 width2 height2]);
        
        % Perform Triangular Erode and Dilate Difference on Subregion
        % in order to highlight high-texture areas
        se = [1 0 0; 1 0 0; 1 1 1];
        eroded = imerode(cropped,se);
        dilated = imdilate(cropped,se);
        g = max(dilated-eroded,0);
%         figure;imshow(g/(max(max(g))));
        % Use the matlab standard deviation filter to highlight 'variety'
        filtered = stdfilt(g)/max(max(stdfilt(g)));
        figure('Name','Possible Segmentation');imshow(filtered);
%         std2(filtered)

        % Subimages with more edges should have more bright pixels,
        % so the mean pixel intensity of the filtered image indicates
        % the relative complexity of the image
        std_mean_list(r,1) = mean(mean(filtered));
    end
end

% The ear region should have the most complexity compared to cheek, 
% neck, or hair, so select the subimage with the most estimated complexity
[max index] = max(std_mean_list);
selected_segmentation = cropped_list{index};
figure('Name','Selected Segmentation');imshow(selected_segmentation);

imwrite(selected_segmentation,strcat(image_dir,'segmented_ear.png'),'png');

toc

disp('Ear Detection and Segmentation Complete');
