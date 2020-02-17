%% WOLA analysis
mr_ratio = 2;
M = 64;
R = floor(M/mr_ratio);
Lh = M/2;

analysis_window_type =  'sinc'; % 'boxcar'; 
switch analysis_window_type
    case 'sinc'
        w_analysis = sinc(nh/M)/M;
    case 'boxcar'
        w_analysis = rectwin(Lh*2)/(Lh*2);
end


% CREATE INPUT SIGNAL
input_signal_type = 'noisy_sinus'; % 'noise';  %   'sinusoidal';   
s = create_input_signal(input_signal_type);


% Transform
S = my_stft(s,w_analysis,R,M);







