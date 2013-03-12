%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Charles Chen - SID: 810797715
% CSCI5722 - Computer Vision
% Final Project - Template Matching Demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Based on the work of Hae Yong Kim and Sidnei Ales de Araujo:
% "Grayscale Template-Matching Invariant to Rotation, Scale, Translation"
% Brightness, and Contrast"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
disp('Begin Template Matching');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup Workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output_dir = 'outputs/templatematch_demo/';
image_dir = 'images/';

disp('Reading in References...');
%target = imread('puzzlepile.png');
target = imread(strcat(image_dir,'transformed_ear.png'));
target_bw = double(target);
figure('Name','Input Image');imshow(target_bw);

temp = imread('images/template_m2.png');
temp_bw = rgb2gray(temp);clear temp;

% Image Grayscale values need to be between [0 1]
temp_bw = cast(temp_bw,'double')/255.0;
% figure('Name','Input Image');imshow(temp_bw);

% Set Up Processing Constants
radii_list = [0 2 3 4 6 8 9 10 12 14 15]';
scale_list = [0.5 0.6 0.7 0.8 0.9 1.0 1.1]';
angle_list = ((5:5:360)-180)'*pi/180;


% Define Thresholds
t1 = 0.80;
tb1 = 0.001;
tg1 = 1.0;

t2 = 0.50;
tb2 = 0.001;
tg2 = 1.0;

t3 = 0.4;

% Blur the template and the target image
gauss = fspecial('gaussian',5,5);
template = imfilter(temp_bw,gauss);
template_mask = temp_bw ~= 0;
test_image = imfilter(target_bw,gauss);
test_image = test_image/max(max(test_image));
% figure;imshow(test_image);

disp('Complete');disp(' ');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Circular Sampling Filter to First Grade Candidates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Begin Circular Filter Sampling');

disp('Item: Sample The Template');
tic
Cq = zeros(size(scale_list,1),size(radii_list,1));
for j=1:size(scale_list,1)
    scaled_template = imresize(template,scale_list(j));
    for i=1:size(radii_list,1)
        post_filter = conv2(scaled_template,Gen_Cifi(radii_list(i)),...
            'same');
        Cq(j,i) = post_filter(round(size(scaled_template,2)/2),...
            round(size(scaled_template,2)/2));
    end
    clear scaled_img;
end
toc


disp('Item: Sample The Image');
tic
% Sample the Image and Calculate Correlations
% Format is y direction, x direction, radii list
Ca = zeros(size(test_image,1),size(test_image,2),size(radii_list,1));
for i=1:size(radii_list,1)
    Ca(:,:,i) = conv2(test_image,Gen_Cifi(radii_list(i)),'same');
end
toc

% Calculate Correlations between Template and Target
disp('Item: Correlate Template and Target');
tic
CisCorr_AQ = zeros(size(test_image,1),size(test_image,2));
CisPS_AQ = zeros(size(test_image,1),size(test_image,2));
Corr_list = zeros(size(scale_list,1),1);
for y=1:size(test_image,1)
    for x=1:size(test_image,2)
        % Calculate Correlation for that Pixel, per Scale
        for i=1:size(scale_list,1)
            Corr_list(i) = BC_Corr(Cq(i,:)',...
                reshape(Ca(y,x,:),size(Ca,3),1),...
                tb1,tg1);
        end
        [maximum index] = max(Corr_list);
        CisCorr_AQ(y,x) = maximum;
        CisPS_AQ(y,x) = index;
    end
end
clear Corr_list maximum index;
toc
disp('Complete');


% Find First Grad Candidate Pixels
% figure;imshow(CisCorr_AQ);
[y x] = find(abs(CisCorr_AQ) >= t1);

% Exclude Results outside of a 15 pixel edge margin
rows = find(y<=15);
y(rows) = [];
x(rows) = [];
rows = find(y>=(size(test_image,1)-15));
y(rows) = [];
x(rows) = [];

rows = find(x<=15);
y(rows) = [];
x(rows) = [];
rows = find(x>=(size(test_image,2)-15));
y(rows) = [];
x(rows) = [];

CisPS_AQ1 = CisPS_AQ(y,x);
CisPS_AQ1 = CisPS_AQ1(:,end);
First_Grade = [y x];
First_Grade_Results = figure('Name','First Grade Candidate Pixels');
imshow(target_bw)
hold on;
scatter(First_Grade(:,2),First_Grade(:,1),'.','r')
axis([1 size(test_image,2) 1 size(test_image,1)]);
set(gca,'YDir','reverse');
hold off;
saveas(First_Grade_Results,...
    strcat(output_dir,'TempMatch_1st_Grade_Results.png'));
disp(strcat('Number of 1st Grade Candidates Found: ',...
    num2str(size(First_Grade,1))));disp(' ');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Radial Sampling Filter to First Grade Candidates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Begin Radial Filter Sampling');

disp('Item: Sample The Template');
% Sample the Template
tic
Rq = zeros(size(angle_list,1),1);
for i=1:size(angle_list,1)
    post_filter = conv2(template,Gen_Rafi(radii_list(end),angle_list(i)),...
            'same');
    Rq(i) = post_filter(round(size(template,2)/2),...
            round(size(template,2)/2));
end
toc

% Sample the Image
disp('Item: Sample The Image');
tic
Ra = zeros(size(test_image,1),size(test_image,2),...
    size(angle_list,1),size(scale_list,1));
for i=1:size(angle_list,1)
    for j=1:size(scale_list,1)
        Ra(:,:,i,j) = conv2(test_image,...
           Gen_Rafi(round(radii_list(end)/scale_list(j)),angle_list(i)),...
            'same');
    end
end

fg_count = size(First_Grade,1);
Ra_list = zeros(fg_count,size(angle_list,1));

for i=1:fg_count
    y = First_Grade(i,1);
    x = First_Grade(i,2);
    for j=1:size(angle_list,1)
        Ra_list(i,j) = Ra(y,x,j,CisPS_AQ(y,x));
    end
end
toc

disp('Item: Correlate');
% Perform Correlation
RasCorr_AQ = zeros(fg_count,1);
RasAng_AQ = zeros(fg_count,1);
Corr_list = zeros(size(angle_list,1),1);
tic
for i=1:fg_count
    for j=1:size(angle_list,1)
       Corr_list(j) = BC_Corr(Ra_list(i,:)',circshift(Rq,j-1),tb2,tg2);
    end
    [maximum index] = max(Corr_list);
    RasCorr_AQ(i) = maximum;
    RasAng_AQ(i) = index;
end
clear Corr_list maximum index;
toc
disp('Complete');

% Find Second Grade Candidate Pixels
% Map the Radial Correlations of the First Grade Candidates
RasCorr_AQ_R = zeros(size(test_image,1),size(test_image,2));
for i=1:fg_count
    y = First_Grade(i,1);
    x = First_Grade(i,2);
    RasCorr_AQ_R(y,x) = RasCorr_AQ(i);
end
% figure;imshow(RasCorr_AQ_R);

row = find(abs(RasCorr_AQ) > t2);
CisPS_AQ2 = CisPS_AQ1(row);
RasAng_AQ2 = RasAng_AQ(row);

Second_Grade = First_Grade(row,:);
Second_Grade_Results = figure('Name','Second Grade Candidate Pixels');
imshow(target_bw)
hold on;
scatter(Second_Grade(:,2),Second_Grade(:,1),'.','r')
axis([1 size(test_image,2) 1 size(test_image,1)]);
set(gca,'YDir','reverse');
hold off;
saveas(Second_Grade_Results,...
    strcat(output_dir,'TempMatch_2nd_Grade_Results.png'));

disp(strcat('Number of 2nd Grade Candidates Found: ',...
    num2str(size(Second_Grade,1))));disp(' ');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Template Matching Filter to Second Grade Candidates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Begin Template Matching Correlation');
tic
sg_count = size(Second_Grade,1);
TempCorr_AQ = zeros(sg_count,1);
for i=1:sg_count
    y = Second_Grade(i,1);
    x = Second_Grade(i,2);
    approx_temp = imrotate(imresize(template,...
        scale_list(CisPS_AQ2(i))),...
        angle_list(RasAng_AQ2(i))*180/pi + 180);
    ymin = y - floor((size(approx_temp,1)-1)/2);
    xmin = x - floor((size(approx_temp,2)-1)/2);
    ymax = y + ceil((size(approx_temp,1)-1)/2);
    xmax = x + ceil((size(approx_temp,2)-1)/2); 
    if ((ymin >0)&&(xmin>0)&&...
            (ymax<=size(test_image,1))&&(xmax<=size(test_image,2)))
        sub_img = test_image(ymin:ymax,xmin:xmax);
        sub_img = reshape(sub_img,...
            size(sub_img,1)*size(sub_img,2),1);
        temp = reshape(approx_temp,...
            size(approx_temp,1)*size(approx_temp,2),1);
        TempCorr_AQ(i) =  BC_Corr(sub_img,temp,tb2,tg2);
    end
end
toc
disp('Complete');

% Map the Correlations of the Matches
TempCorr_AQ_R = zeros(size(test_image,1),size(test_image,2));
for i=1:sg_count
    y = Second_Grade(i,1);
    x = Second_Grade(i,2);
    TempCorr_AQ_R(y,x) = TempCorr_AQ(i);
end
% figure;imshow(TempCorr_AQ_R);

% Display Results
row = find(abs(TempCorr_AQ) >= t3);
CisPS_AQ3 = CisPS_AQ2(row);
RasAng_AQ3 = RasAng_AQ2(row);

Third_Grade = Second_Grade(row,:);
Third_Grade_Results = figure('Name','Third Grade Candidate Pixels');
imshow(target_bw)
hold on;
scatter(Third_Grade(:,2),Third_Grade(:,1),'.','r')
axis([1 size(test_image,2) 1 size(test_image,1)]);
set(gca,'YDir','reverse');
hold off;
saveas(Third_Grade_Results,...
    strcat(output_dir,'TempMatch_3rd_Grade_Results.png'));

disp(strcat('Number of 3rd Grade Candidates Found: ',...
    num2str(size(Third_Grade,1))));disp(' ');

% Select Best match of the Group
index = find(abs(TempCorr_AQ) == max(abs(TempCorr_AQ)));
Best = Second_Grade(index,:);
Best_Match = figure('Name','Point of Best Correlation');
imshow(target_bw)
hold on;
scatter(Best(:,2),Best(:,1),'.','r')
axis([1 size(test_image,2) 1 size(test_image,1)]);
set(gca,'YDir','reverse');
hold off;
saveas(Best_Match,...
    strcat(output_dir,'TempMatch_4_Best_Match.png'));


disp('Template Matching Complete');
disp(strcat('Overall Template to Image Correlation: ',...
    num2str(TempCorr_AQ(index))));
