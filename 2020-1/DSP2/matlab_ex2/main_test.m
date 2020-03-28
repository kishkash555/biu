% Short-Time Fourier Transform
% Demo Program
% Digital Signal Processing II (83-624), Bar-Ilan University
% Revised 7 January, 2014
% Sharon Gannot
%% Sample file

close all
clear all
clc
[s,fs] = audioread('mewm0-si718.wav'); % 'medr0-si744' % 'mewm0-si718' % 'fdmy0-sx297'
%s = cos(2*pi/256*97*(00:9999)); fs = 8000; 
%s = exp(2*pi*sqrt(-1)/256*97*(00:9999)); fs = 8000;

if size(s,1)<size(s,2),s=s';end
figure(1)
plot([0:length(s)-1]/fs,s)
title('Original Signal','fontsize',14);
xlabel('Time[Sec]','fontsize',14);
ylabel('Amplitude','fontsize',14);
set(gca,'fontsize',14);
soundsc(s,fs)
%% Analysis

Lh = 512; %256;%1024;15; % Window length
% NFFT - FFT length. 


Synth_Method = 'No_Overlap'; % 'WOLA'; 'FBS'; 'No_Overlap'
switch Synth_Method
    case 'WOLA'
        NFFT = Lh;
        w_analysis =  hamming(Lh); %hamming(Lh); boxcar(Lh);
        R = fix(Lh/4); % Jump 
    case 'FBS'
        NFFT = Lh/4;
        w_analysis =  sinc((-Lh/2:Lh/2-1)/NFFT)'/NFFT; % boxcar(Lh); 
        w_analysis = w_analysis.*hamming(length(w_analysis)); % Using hamming to reduce sidelobe level
        R = 1; % Jump
    case 'No_Overlap'
        NFFT = Lh;
        w_analysis = hamming(Lh); 
        R = Lh; % Jump   
end;

S = stft(s, w_analysis, R ,NFFT);
S1 = my_stft(s, w_analysis, R, NFFT);

sprintf("S: (%d,%d) S1: (%d,%d), diff: %.3f",size(S), size(S1), norm(S-S1))
%% Sonogram

T = (0:size(S,2))/fs*R;
F = (0:NFFT/2)*fs/2/(NFFT/2);
figure(2)
imagesc(T,F,20*log10(abs(S(1:NFFT/2+1,:)+eps)))
axis xy
xlabel('Time[Sec]','fontsize',14);
ylabel('Frequency[Hz]','fontsize',14);
set(gca,'fontsize',14);
colorbar

%% Weighted Overlap and Add - Synthesis

Synth_Win = 'No_Overlap'; % 'Perfect' ; 'Rectangular' ; 'No_Overlap'
switch Synth_Win
    case 'Rectangular'
        Lf = Lh;
        w_synthesis = boxcar(Lf);
    case 'Perfect'
        w_synthesis = synthesis_win(w_analysis, R); % In this program Lf = Lh;
    case 'No_Overlap'
        w_synthesis = 1./w_analysis; % By Definition Lf = Lh;
end
s_hat = real(istft(S, w_synthesis, R));
s_hat1 = real(my_istft(S, w_synthesis, R));
norm(s_hat-s_hat1)
%% Filter Bank Sum - Synthesis

s_hat = FBS(S);

%% Calculate the window gain (for WOLA with boxcar synthesis window)
w = zeros(Lh*5,1);
w(3*Lh+1:4*Lh) = w_analysis;
Seg_No = 1+floor((length(w)-Lh)/R);

W = zeros(Seg_No,Lh);
for p = 1:Seg_No;
    W(p,:) = w((p-1)*R+1:(p-1)*R+Lh)';
end;
ww = sum(W);
win_gain = mean(ww);
D = 1;
ERR_win = 10*log10(sum((1-ww/win_gain).^2));

figure(3),plot(ww)
title(['Overlapping windows (window error = ',num2str(ERR_win,4),'dB)'],'fontsize',14);
xlabel('Sample Number','fontsize',14);
ylabel('Amplitude','fontsize',14);
set(gca,'fontsize',14);

%% Calculate the window gain and delay (for WOLA with perfect/no_overlap synthesis window)

win_gain = 1;
D = 1;

%% Calculate the window gain and delay (for FBS)

[win_gain,D] = max(w_analysis);

%% Test reconstructed signal
LL = min(length(s),length(s_hat)); % For presenting the reconstructed signal
tt = (Lh-R)+1:LL-Lh;

s_d = delay(s_hat,D-1);
SNR_rec=10*log10(var(s(tt))/var(s(tt)-s_d(tt)/win_gain));
figure(4)
subplot(211)
plot(tt/fs,s_d(tt)/win_gain)
title('Reconstructed Signal','fontsize',14);
xlabel('Time[Sec]','fontsize',14);
ylabel('Amplitude','fontsize',14);
set(gca,'fontsize',14);
subplot(212)
plot(tt/fs,s(tt)-s_d(tt)/win_gain)
title(['Reconstruction Error (SNR=',num2str(SNR_rec,4),'dB)'],'fontsize',14);
xlabel('Time[Sec]','fontsize',14);
ylabel('Amplitude','fontsize',14);
set(gca,'fontsize',14);

soundsc(s_d)
%% Filtering - Prepare the filter
Lg = 75;
Hd = LPF(Lg-1);
g = Hd.Numerator'; % [zeros(15,1); 1 ; 0];

figure(5)
stem(0:Lg-1,g)
figure(6)
[gg,omega] = freqz(g,1,2048);
plot(omega/pi*fs/2,10*log10(abs(gg)));
title('Frequency Response','fontsize',14);
xlabel('Frequency[Hz]','fontsize',14);
ylabel('Amplitude[dB]','fontsize',14);
set(gca,'fontsize',14);
s_fil_conv = filter(g,1,s);
soundsc(s_fil_conv)
%% : Overlap & Add
NFFT = 256;
G = fft(g,NFFT,1);

Lh = NFFT-Lg+1; % Analysis window length
R = Lh;
w_analysis = boxcar(Lh); % In OLA we use only Boxcar windows
Lf = NFFT; % Synthesis window length 

% It is assumed that FFT size is smaller or equal the synthesis window. 
%If this is not the case, it is the user responsibility to use correct
%zero-padding of the windows to the correct FFT size.

w_synthesis = boxcar(Lf);  % In OLA we use only Boxcar windows


S = stft(s, w_analysis, R ,NFFT);
S_FIL = repmat(G,1,size(S,2)).*S; % Frequency band processing 
s_fil_ola = real(istft(S_FIL, w_synthesis, R));


LL = min(length(s),length(s_fil_ola)); %For presenting the reconstructed signal
tt = (Lh-R)+1:LL-Lh;

SNR_fil_ola = 10*log10(var(s_fil_conv(tt))/var([s_fil_conv(tt)-s_fil_ola(tt)]));

figure(7)
plot([ s_fil_conv(tt)-s_fil_ola(tt)])
title(['Overlap Add Reconstruction Error (SNR=',num2str(SNR_fil_ola,4),'dB)'],'fontsize',14);

xlabel('Time[Sec]','fontsize',14);
ylabel('Amplitude','fontsize',14);
set(gca,'fontsize',14);

soundsc(s_fil_ola)

%% : Overlap & Save
% The parameters are wrong !!!
% % % (Overlap-save algorithm for linear convolution)
% % % h = FIR_impulse_response
% % % M = length(h)
% % % overlap = M − 1
% % % N = 8 × overlap    (see next section for a better choice)
% % % step_size = N − overlap
% % % H = DFT(h, N)
% % % position = 0
% % % 
% % % while position + N ≤ length(x)
% % %     yt = IDFT(DFT(x(position+(1:N))) × H)
% % %     y(position+(1:step_size)) = yt(M : N)    (discard M−1 y-values)
% % %     position = position + step_size
% % % end

% original code
NFFT = 256;

G = fft(g,NFFT);

Lh = NFFT; % Analysis window length
% Original code
% R = Lg+1; 
% corrected
R = NFFT- Lg + 1;

w_analysis = boxcar(Lh);
% original code
% Lf = Lg; % Synthesis window length
% corrected
Lf = NFFT;

% It is assumed that FFT size is smaller or equal the synthesis window. 
% If this is not the case, it is the user responsibility to use correct
% zero-padding of the windows to the correct FFT size.
% original
% w_synthesis = zeros(size(w_analysis));

w_synthesis = zeros(Lf,1);
% original
% w_synthesis(1:Lg) = boxcar(Lg);
% corrected
w_synthesis(Lg:end) = boxcar(Lf-Lg+1);


S = stft(s, w_analysis, R ,NFFT);
S_FIL = repmat(G,1,size(S,2)).*S; % Frequency band processing 
s_fil_ols = real(istft(S_FIL, w_synthesis, R));

LL = min(length(s),length(s_fil_ols)); %For presenting the reconstructed signal
tt = (Lh-R)+1:LL-Lh;

SNR_fil_ols = 10*log10(var(s_fil_conv(tt))/var([s_fil_conv(tt)-s_fil_ols(tt)]));

figure(7)
plot([ s_fil_conv(tt)-s_fil_ols(tt)])
title(['Overlap & Save Reconstruction Error (SNR=',num2str(SNR_fil_ols,4),'dB)'],'fontsize',14);

xlabel('Time[Sec]','fontsize',14);
ylabel('Amplitude','fontsize',14);
set(gca,'fontsize',14);

soundsc(s_fil_ols)
%% Filtering with WOLA

Lh = 512; %256;%1024;15; % Window length
NFFT = Lh;
w_analysis =  hamming(Lh); % hamming(Lh); boxcar(Lh);
R = fix(Lh/4); % Jump
        
w_synthesis = synthesis_win(w_analysis, R);

S = stft(s, w_analysis, R ,NFFT);


     
G = fft(g,NFFT,1);

S_FIL = repmat(G,1,size(S,2)).*S;
s_fil_wola = real(istft(S_FIL, w_synthesis, R));

LL = min(length(s_fil_wola),length(s_fil_conv)); % For presenting the reconstructed signal
tt = (Lh-R)+1:LL-Lh;

SNR_fil_rec = 10*log10(var(s_fil_conv(tt))/var([s_fil_conv(tt)-s_fil_wola(tt)]));

figure(8)
plot(tt/fs, s_fil_conv(tt),tt/fs, s_fil_wola(tt))


title(['STFT Reconstruction Error (SNR=',num2str(SNR_fil_rec,4),'dB)'],'fontsize',14);


xlabel('Time[Sec]','fontsize',14);
ylabel('Amplitude','fontsize',14);
set(gca,'fontsize',14);

soundsc(s_fil_wola)
