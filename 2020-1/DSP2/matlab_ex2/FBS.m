function [s_hat] = FBS(S)

NFFT = size(S,1);
k = 0:NFFT-1;
Seg_No = size(S,2); 
n = 0:Seg_No-1;
[nn,kk] = meshgrid(n,k);

FF = exp(j*2*pi/NFFT*kk.*nn);

S = S.*FF;

s_hat = real(sum(S))'/NFFT;