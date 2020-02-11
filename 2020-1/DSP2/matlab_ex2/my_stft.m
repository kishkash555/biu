function S = my_stft(s, w_analysis, R ,NFFT)
Ls = length(s);
Lh = length(w_analysis);
n_slices = floor((Ls-Lh)/R)+1;

if Ls < Lh
    S = zeros(0,0);
    return
end    
all_windows = zeros(NFFT,n_slices);
for m = 0:n_slices-1
    slice = s(m*R+1: m*R+Lh);
    windowed_slice = slice .* w_analysis;
    if Lh > NFFT
        folded_y = reshape(windowed_slice,NFFT,[]);
        folded_y = sum(folded_y,2);
        all_windows(:, m+1) = folded_y;
    else
        all_windows(1:Lh, m+1) = windowed_slice;
    end
end

S = fft(all_windows);

n = (0:n_slices-1)*R;
k = (0:NFFT-1)';
baseband = (2*pi/NFFT)*k*n;

S = S .* exp(-1i*baseband);


    
    