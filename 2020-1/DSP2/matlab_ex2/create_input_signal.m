function s = create_input_signal(signal_type,n, P)
k1 = 12; % an arbitrary frequency for the input signal
if n == 0
    n=1024*32;
end
ns = (0:(n-1))'; % the index _n_ for the signal

switch signal_type
    case 'sinusoidal'
        s = cos(2*pi*k1*ns/P); % the signal
    case 'noisy_sinus'
        s = cos(2*pi*k1*ns/P) + 0.2*randn(size(ns));
    case 'noise'
        s = randn(size(ns));
end

end