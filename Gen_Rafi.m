%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Radial Sampling Filter Generator
% Charles Chen
% Computer Vision Fall 2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate a r by r matrix for use as a Rafi Filter
% of radius r, angle alpha.  Non-Zero cells are scaled to one over the
% number non-zero cells in the matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ Rafi ] = Gen_Rafi(r,alpha)
    % Alpha is the range of (-pi pi]
    if (r == 0)
        Rafi = [1];
    else
        [px,py] = meshgrid(1:(2*r+1), 1:(2*r+1));
        py = flipud(py);
        boundary1 =(sqrt((px-(r+1)).^2 + (py-(r+1)).^2) <= r);
        boundary2 = (abs(atan2((py-(r+1)),(px-(r+1))) - alpha)...
            <= (3*pi/180));
        
%         boundary3 =(sqrt((px-(r+1)).^2 + (py-(r+1)).^2) < max(r/2,1));
%         boundary4 = (abs(atan2((py-(r+1)),(px-(r+1))) - alpha)...
%             <= (8*pi/180));
        
        radial_mask = (boundary1 & boundary2);% | (boundary3 & boundary4);
        radial_mask(r+1,r+1) = 1;
        
        [cx cy] = find(radial_mask ~= 0);
        Rafi = radial_mask / max(1,size(cx,1));
    end
end
