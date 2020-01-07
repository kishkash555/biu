function y = resample(x,L,M)
    %RESAMPLE Summary of this function goes here
    %   Detailed explanation goes here
    h = create_lpf(L,M);
    g = fir_to_pp(L,M,h);
    y = run_pp_filter(x,g,L,M);
end

