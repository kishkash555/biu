
L=5; 
M=7;
x = audioread("doors.wav");
h = create_lpf(L,M);
g = fir_to_pp(L,M,h);
y = run_pp_filter(x,g,L,M);
soundsc(y,8000*L/M);



