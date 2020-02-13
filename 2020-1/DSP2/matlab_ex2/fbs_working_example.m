%% STFT of a single-frequency signal

% Analysis filter half-size 
Lh = 1024;
% the index _n_ for creating the analysis window
nh=(-Lh:Lh-1)';

% DFT length
M=64;

% ANALYSIS FILTER
% if we wanted to calculate the sinc from the sine function: 
%%% w_analysis = sin(pi*nh/M)./(pi*nh);
%%% w_analysis(1025)=1/M;

% using matlab's signal processing toolbox sinc function:
w_analysis = sinc(nh/M)/M;


% CREATE INPUT SIGNAL
k1 = 12; % an arbitrary frequency for the input signal
ns = (0:(1024*32-1))'; % the index _n_ for the signal
s = cos(2*pi*k1*ns/M); % the signal


% STFT
s_zeropad = [zeros(Lh,1); s; zeros(Lh-1,1)]; % the analysis filter "eats up" 2*Lh-1 samples from the output
S = my_stft(s_zeropad,w_analysis,1,M);
t = abs(S);
figure();
h=pcolor(t(1:64,:));
set(h,'EdgeColor','None')

%% FBS syntesis (requires previous cell)

% calcualte the baseband factor
[NFFT, n_slices] = size(S);
n = (0:n_slices-1);
k = (0:NFFT-1)';
% since _k_ is column and _n_ is row, 
% multiplying them will get their cartesian-product matrix:
baseband = (2*pi/NFFT)*k*n; 

% We are not running the istft function.
% Instead, preform the inverse baseband shift 
% and sum the STFT coefficients:
Sbb = S .* exp(1i*baseband); % shift back from baseband

r = sum(Sbb)'; % recounstructed signal


% RESULTS
% check the signal is real
sprintf('imaginary part: %.2f dB', ...
    10*log10( ... 
        norm(imag(r)) / norm(real(r)) ...
        )...
    )
    
sprintf('Reconstruction error: %.2f dB',10*log10(norm(r-s)/norm(s)))
% plot the reconstructed signal on the original
figure();
plot(real(r(1:50)),'b-')
hold on;
plot(s(1:50),'r-')


