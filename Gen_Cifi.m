%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Circular Sampling Filter Generator
% Charles Chen
% Computer Vision Fall 2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate a r by r matrix for use as a Cifi Filter
% of radius r.  Non-Zero cells are scaled to one over the number
% non-zero cells in the matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ Cifi ] = Gen_Cifi(r)
    if (r == 0)
        Cifi = [1];
    else
        [px,py] = meshgrid(1:(2*r+1), 1:(2*r+1));

        boundary1 =(sqrt((px-(r+1)).^2 + (py-(r+1)).^2) <= r);
        boundary2 =(sqrt((px-(r+1)).^2 + (py-(r+1)).^2) > r - 1);
        circle_mask = boundary1 & boundary2;
        
        [cx cy] = find(circle_mask ~= 0);
        Cifi = circle_mask / size(cx,1);
    end
end

