%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Charles Chen - SID: 810797715
% CSCI5722 - Computer Vision
% Final Project - Force Field Transform Demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Based on the work of Hurley, David J., Mark S. Nixon, John N. Carter:
% "Force field Energy Functionals for image feature extraction"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
disp('Begin Force Field Transform');

output_dir = 'outputs/forcefield_demo/';
image_dir = 'images/';

% segmented_ear.png is assumed to be a grayscale image
bw = imread(strcat(image_dir,'segmented_ear.png'));
figure('Name','Input Image');imshow(bw);
% Establish Sizing Constants
[IY IX] = size(bw);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Invertible Linear Transform
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize Distance Vector Matrix
r_matrix = zeros(IY,IX,2);
for iy=1:IY
   for ix=1:IX
      r_matrix(iy,ix,:) = [iy ix];
   end
end

P_list = cast(reshape(bw,IX*IY,1),'double');
p_count = size(P_list,1);
r_list = reshape(r_matrix,IX*IY,2);

df_matrix = zeros(p_count,p_count,2);   % d matrix for force field calcs
de_matrix = zeros(p_count,p_count);     % d matrix for energy field calcs

disp('Begin d matrix initialization');
tic;
% This step can be pre-processed and saved if a fixed input image size
% is assumed.
% Outer loop goes across columns
for j=1:p_count
    % Inner loop goes through rows, down to the diagnal
    % Zeros along the diagnal, so dont need to edit those
    for i=1:(j-1)
        % Calculate d_ij for force calculation
        df_ij = (r_list(j,:) - r_list(i,:)) /...
            (norm(r_list(j,:) - r_list(i,:)))^3;
        df_matrix(j,i,:) = df_ij;
        % Assign the negative value of this to the
        % corresponding cell opposite the diagnal
        df_matrix(i,j,:) = -df_ij;
        
        % calculate d_ij for energy calculation
        de_ij = 1 / norm(r_list(j,:) - r_list(i,:));
        de_matrix(j,i) = de_ij;
        de_matrix(i,j) = de_ij;
    end
end
toc;
disp('Complete');


disp('Begin Force Field Calculation');
tic;
% Generate Force Vector List (one per directional component)
F_x = df_matrix(:,:,1) * P_list;
F_y = df_matrix(:,:,2) * P_list;
% Merge Components and reshape into image dimensions
F_list(:,1) = F_x;
F_list(:,2) = F_y;
F_matrix = reshape(F_list,IY,IX,2);
toc;
disp('Complete');

disp('Begin Energy Field Calculation');
tic;
% Generate Energy value List
E_list = de_matrix(:,:) * P_list;
% Reshape into image dimensions
E_matrix = reshape(E_list,IY,IX);
toc;
disp('Complete');


% Plot Vector Field over Energy Field
[g_x,g_y] = meshgrid(1:IX,1:IY);
F_z =  zeros(IY,IX);
force_field = figure('Name','Force Field');
mesh(g_x,g_y,E_matrix);
axis tight;
hold on;
quiver3(g_x,g_y,E_matrix,F_matrix(:,:,2),F_matrix(:,:,1),F_z,0.05)
set(gca,'YDir','reverse');
view(0,90);
axis equal;
hold off;
saveas(force_field,strcat(output_dir,'Force_03_ForceField.png'),'png') 
% % Optional: Plot Energy Field As Well
% figure;mesh(g_x,g_y,E_matrix.*E_matrix);
% figure;imshow(E_matrix/max(max(E_matrix)));




% Display Force Field Magnitudes
F_mag =  zeros(IY,IX);
for x=1:IX
   for y=1:IY
      F_mag(y,x) = norm([F_matrix(y,x,1) F_matrix(y,x,2)]);
   end
end
scaled_F_mag = F_mag/(max(max(F_mag)));
% Display the Force (Magnitude) Field
figure('Name','Force Field Magnitudes'); imshow(scaled_F_mag);
imwrite(scaled_F_mag,...
    strcat(output_dir,'Force_04_Fmag.png'),'png');



% Plot Divergence of Vector Field
F_div = divergence(g_x,g_y,F_matrix(:,:,2),F_matrix(:,:,1));
F_div_scaled = F_div/max(max(F_div));
figure('Name','Divergence of Force Field');imshow(F_div_scaled);
imwrite(F_div_scaled,...
    strcat(output_dir,'Force_05_Fdiv.png'),'png');
imwrite(histeq(F_div_scaled),...
    strcat(output_dir,'Force_05_Fdiv_contrasted.png'),'png');

F_div_threshed = im2bw(histeq(F_div_scaled), 190./255.);
% Mask out a 6px margin around the center
edge_mask = zeros(size(F_div_threshed,1),size(F_div_threshed,2));
edge_mask(6:end-6,6:end-6) = 1;
F_div_threshed = F_div_threshed .* edge_mask;
figure('Name','Thresholded Divergence Field');imshow(F_div_threshed);
imwrite(F_div_threshed,...
    strcat(output_dir,'Force_05_Fdiv_masked.png'),'png');
imwrite(F_div_threshed,...
    strcat(image_dir,'transformed_ear.png'),'png');

% Optional: Detect Local Minima in Energy Field
% % Find Local Minima in Energy Field
% E_min_mask = find_local_min(E_matrix);
% figure;imshow(E_min_mask);

