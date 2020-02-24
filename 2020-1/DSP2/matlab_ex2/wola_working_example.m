%% WOLA analysis
mr_ratio = 8;
M = 64;
R = floor(M/mr_ratio);
Lh = M;
Lf = M; 
w_analysis = rectwin(Lh);
w_synthesis = w_analysis/mr_ratio;

% CREATE INPUT SIGNAL
input_signal_type = 'noisy_sinus';     % 'noise';  %   'sinusoidal';; 
s = create_input_signal(input_signal_type,512,M*5);

% Transform
S = my_stft(s,w_analysis,R,M);
r = my_istft(S,w_synthesis,R);

tt = M: length(r)-M;
sprintf('Reconstruction error: %.2f dB',10*log10(var(r(tt)-s(tt))/var(s(tt))))

figure();
plot(real(s))
hold on
plot(real(r))



  

