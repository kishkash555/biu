function s = my_istft(S, w_synthesis, R)

Lf = length(w_synthesis);
[NFFT, n_slices] = size(S);
Ls = (n_slices-1)*R + Lf;
unfold_factor = floor(w_synthesis/NFFT);

if numel(S)==0
    s = zeros(0,0);
    return
end

n = (0:n_slices-1)*R;
k = (0:NFFT-1)';
baseband = (2*pi/NFFT)*k*n;

S = S .* exp(1i*baseband);
all_ifft = ifft(S);

s = zeros(Ls,1);
buffer = zeros(Lf,1);

for m = 0:n_slices-1
    slice = all_ifft(:,m+1);
    if unfold_factor > 1
        slice = repmat(slice, unfold_factor,1);
    end
    windowed_slice = slice .* w_synthesis;
    buffer = buffer + windowed_slice;
    s(m*R+1 : m*R+Lf) = buffer(1:Lf);
    buffer = circshift(buffer, -R);
    buffer(Lf-R+1:Lf)=0;
end

%s(Ls-Lf+NFFT:end)=buffer(1:


    
    