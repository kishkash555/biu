%% no window
x = (1:512)'; 
y = cos(2*pi*0.02*x)+4*sin(2*pi*0.034*x);
Y = fft(y);
Y1 = stft(y,ones(512,1),512,512);
norm(Y-Y1)
% 0

%% with window
sprintf("with window")
x = (1:512)'; 
y = cos(2*pi*0.02*x)+4*sin(2*pi*0.034*x); 
w_analysis =  hamming(512);
Y = fft(y .* w_analysis);
Y1 = stft(y,w_analysis,512,512);
norm(Y-Y1)
% 0

%% longer signal, dft on each part separately
sprintf("dft on each part separately")

x = (1:512)'; 
y = [cos(2*pi*0.02*x)+4*sin(2*pi*0.034*x); cos(2*pi*0.014*x)+4*sin(2*pi*0.024*x)]; 
w_analysis =  hamming(512);
Y1 = stft(y,w_analysis,512,512);

Y = fft(y(1:512) .* w_analysis);
norm(Y-Y1(:,1))

Y = fft(y(513:end) .* w_analysis);
norm(Y-Y1(:,2))

sprintf("This shoud be a large number")
norm(Y-Y1(:,1))

%% one slice, wrong folding scheme
sprintf("with folding")
x = (1:512)'; 
y = cos(2*pi*0.02*x)+4*sin(2*pi*0.034*x); 
% w_analysis =  hamming(512);
folded_y = reshape(y,256,2);
folded_y = sum(folded_y');


Y = fft(folded_y)';
Y1 = stft(y,ones(256,1),256,256);
norm(Y-Y1)
figure();plot(abs(Y),'r-');hold on; plot(abs(Y1)+50,'b-')
figure();plot(angle(Y),'r-');hold on; plot(angle(Y1),'b-')


%% one slice, with correct folding
sprintf("with other folding")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1711)*100;

% w_analysis =  hamming(512);
folded_y = reshape(y,256,2);
folded_y = sum(folded_y')';

Y = fft(folded_y);
Y1 = stft(y,ones(512,1),256,256);

figure();plot(abs(Y),'r-');hold on; plot(abs(Y1)+10,'b-')
figure();plot(angle(Y),'r-');hold on; plot(angle(Y1),'b-')

norm(Y-Y1)

%% one slice, with correct folding and window
sprintf("with other folding")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1711)*100;

w_analysis =  hamming(512);
folded_y = reshape(y .* w_analysis ,256,2);
folded_y = sum(folded_y')';

Y = fft(folded_y);
Y1 = stft(y,w_analysis,256,256);

figure();plot(abs(Y),'r-');hold on; plot(abs(Y1)+10,'b-')
figure();plot(angle(Y),'r-');hold on; plot(angle(Y1),'b-')

norm(Y-Y1)

%% same, longer signal, compare each part without overlap
sprintf("with other folding")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+1024)*100;

w_analysis =  hamming(512);
folded_y = reshape(y(513:1024) .* w_analysis ,256,2);
folded_y = sum(folded_y')';

Y = fft(folded_y);
Y1 = stft(y,w_analysis,512,256);

figure();plot(abs(Y),'r-');hold on; plot(abs(Y1(:,2))+10,'b-')
figure();plot(angle(Y),'r-');hold on; plot(angle(Y1(:,2)),'b-')

norm(Y-Y1(:,2))

%% same, longer signal, compare each part with overlap
sprintf("with folding and overlap")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+1024)*100;
w_analysis =  hamming(512);
Y1 = stft(y,w_analysis,200,256);

% compare first slice
folded_y = reshape(y(1:512) .* w_analysis ,256,2);
folded_y = sum(folded_y')';
Y = fft(folded_y);
norm(Y-Y1(:,1))

% compare second slice
folded_y = reshape(y(201:712) .* w_analysis ,256,2);
folded_y = sum(folded_y')';
Y = fft(folded_y);
k=(0:255)'; M = 256; n=200;
baseband = 2*pi*k*n/M;
bb_phase = exp(-1i*baseband);
Y = Y .* bb_phase;
norm(Y-Y1(:,2))

% compare third slice - padding zeros
folded_y = reshape(y(401:912) .* w_analysis ,256,2);
folded_y = sum(folded_y')';
Y = fft(folded_y);
k=(0:255)'; M = 256; n=400;
baseband = 2*pi*k*n/M;
bb_phase = exp(-1i*baseband);
Y = Y .* bb_phase;
norm(Y-Y1(:,3))

figure();plot(abs(Y),'r-');hold on; plot(abs(Y1(:,3))+0.1,'b-')
figure();plot(angle(Y),'r-');hold on; plot(angle(Y1(:,3)),'b-')


%% FFT length doesn't match window length
sprintf("window-fft size mismatch")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+1024)*100;
w_analysis =  hamming(500);
Y1 = stft(y,w_analysis,200,256);

% compare first slice
folded_y = zeros(512,1);
folded_y(1:500) = y(1:500);

folded_y = reshape(folded_y .* w_analysis ,256,2);
folded_y = sum(folded_y')';
Y = fft(folded_y);
norm(Y-Y1(:,1))

%% test the implementation
sprintf("test implementation with folding and overlap")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+1024)*100;
w_analysis =  hamming(512);
Y1 =    stft(y, w_analysis, 200, 256);
Y2 = my_stft(y, w_analysis, 200, 256);
norm(Y1(:,1)-Y2(:,1))
norm(Y1(:,2)-Y2(:,2))
norm(Y1(:,3)-Y2(:,3))

figure();plot(abs(Y1(:,2)),'r-');hold on; plot(abs(Y2(:,2))+0.1,'b-')
figure();plot(angle(Y1(:,2)),'r-');hold on; plot(angle(Y2(:,2))+0.1,'b-')

%% test the implementation - nonsymmetric window
sprintf("test implementation, non symmetric window")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+1024)*100;
w_analysis =  hamming(512);
w_analysis = w_analysis(1:256);
Y1 =    stft(y, w_analysis, 200, 256);
Y2 = my_stft(y, w_analysis, 200, 256);
norm(Y1(:,1)-Y2(:,1))
norm(Y1(:,2)-Y2(:,2))
norm(Y1(:,3)-Y2(:,3))

%% test the implementation - analysis window shorter than NFFT
sprintf("test implementation, non symmetric window")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+1024)*100;
w_analysis =  hamming(128);
Y1 =    stft(y, w_analysis, 200, 256);
Y2 = my_stft(y, w_analysis, 200, 256);
norm(Y1(:,1)-Y2(:,1))
norm(Y1(:,2)-Y2(:,2))
norm(Y1(:,3)-Y2(:,3))


%% test the implementation - signal shorter than analysis window 
sprintf("test implementation, signal shorter than analysis")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+100)*100;
w_analysis =  hamming(128);
Y1 =    stft(y, w_analysis, 200, 256);
Y2 = my_stft(y, w_analysis, 200, 256);
norm(Y1(:,1)-Y2(:,1))
norm(Y1(:,2)-Y2(:,2))
norm(Y1(:,3)-Y2(:,3))

%% test the implementation - signal shorter than analysis window 
sprintf("test implementation, signal longer than analysis but shorter than R")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+150)*100;
w_analysis =  hamming(128);
Y1 =    stft(y, w_analysis, 200, 256);
Y2 = my_stft(y, w_analysis, 200, 256);
norm(Y1(:,1)-Y2(:,1))


%% test the implementation - signal shorter than analysis window 
sprintf("test implementation, signal longer than R but shorter than analysis window")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+100)*100;
w_analysis =  hamming(128);
Y1 =    stft(y, w_analysis, 1, 256);
Y2 = my_stft(y, w_analysis, 1, 256);
norm(Y1(:,1)-Y2(:,1))
