%% test inverse transform
sprintf("basic transform <-> inverse transform")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+511)*100;
w_analysis =  boxcar(512);

Y =    stft(y, w_analysis, 512, 512);

y1 = istft(Y,w_analysis,512);
y2 = my_istft(Y,w_analysis,512);

norm(y1-y2)

%% test inverse transform
sprintf("basic transform <-> inverse transform")
[s,fs] = audioread('mewm0-si718.wav');
y=s(1200:1200+511)*100;
w_analysis =  boxcar(512);

Y =    stft(y, w_analysis, 512, 512);

y1 = istft(Y,w_analysis,512);
y2 = my_istft(Y,w_analysis,512);

norm(y1-y2)

