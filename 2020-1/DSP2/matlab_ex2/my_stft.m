function S = my_stft(s, w_analysis, R ,NFFT)
Ls = length(s);
Lh = length(w_analysis);
n_slices = floor((Ls-Lh+1)/R)+1;


all_windows = zeros(n_slices,NFFT);

for m = 0:n_slices-1
    slice = s(m*R+1: m*R+Lh);
    windowed_slice = slice .* w_analysis;
    folded_y = reshape(windowed_slice,NFFT,[]);
    all_windows(m+1,:) = sum(folded_y')';
end

S = fft(all_windows,NFFT,2)';

n = (0:n_slices-1)*R;
k = (0:NFFT-1)';
baseband = (2*pi/NFFT)*k*n;

S = S .* exp(-1i*baseband);


    
    