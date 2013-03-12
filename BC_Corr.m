%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Brightness/Contrast Aware Correlation
% Charles Chen
% Computer Vision Fall 2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the brightness and contrast aware correlation for
% thresholds tb and tg.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ corr_bc ] = BC_Corr(x,y,tb,tg)
% Correlation that takes into account brightness and contrast,
% as defined in "Grayscale Template-Matching Invariant to Rotation,
% Scale, Translation, Brightness, and Contrast" (Kim, Araujo)

x_tilde = x - mean(x);
y_tilde = y - mean(y);

beta = pinv(x_tilde)*y_tilde ;
beta = beta(:,end);

gamma = mean(y) - beta*mean(x);

if (le(norm(beta),tb)|le(1/tb,norm(beta))|(abs(gamma)>tg))
    corr_bc = 0;
else
    corr_bc = corr(x,y);
end

end

