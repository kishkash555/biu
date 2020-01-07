function y = resample_naive(x,L,M)
    %RESAMPLE_NAIVE Summary of this function goes here
    %   Detailed explanation goes here
    t = length(x);
    x_i = zeros(1,t*L);
    x_i(1,1:L:end) = x;
    hL = create_lpf(L,M);
    x_if = filter(hL,1,x_i);
    x_d = x_if(1:M:end);
    y = x_d;
 end

