%% no window
x = (1:512)'; 
y = cos(2*pi*0.02*x)+4*sin(2*pi*0.034*x);
Y = my_stft(y,ones(512,1),512,512);
Y1 = stft(y,ones(512,1),512,512);
norm(Y-Y1)
% 0

%% with window
sprintf("with window")
x = (1:512)'; 
y = cos(2*pi*0.02*x)+4*sin(2*pi*0.034*x); 
w_analysis =  hamming(512);
Y = my_stft(y,w_analysis,512,512);
Y1 = stft(y,w_analysis,512,512);
norm(Y-Y1)
% 0

%% longer signal, dft on each part separately
sprintf("dft on each part separately")

x = (1:512)'; 
y = [cos(2*pi*0.02*x)+4*sin(2*pi*0.034*x); cos(2*pi*0.014*x)+4*sin(2*pi*0.024*x)]; 
w_analysis =  hamming(512);
Y1 = stft(y,w_analysis,512,512);

Y = my_stft(y(1:512),w_analysis, 512, 512);
norm(Y-Y1(:,1))

Y = my_stft(y(513:end),w_analysis,512,512);
norm(Y-Y1(:,2))

sprintf("This shoud be a large number")
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
