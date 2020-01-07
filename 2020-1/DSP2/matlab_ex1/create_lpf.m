function fir_coeffs = create_lpf(L,M)
    %CREATE_LPF create an FIR LPF for rational rate change
    %   Detailed explanation goes here
    wc = min(1/L,1/M);
    fir_coeffs = L*fir1(200,wc,'low');
end

