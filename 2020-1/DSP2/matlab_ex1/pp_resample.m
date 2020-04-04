function y = pp_resample(x,L,M)

h = create_lpf(L,M);
h = pad_divisible(h,L*M);
x = pad_divisible(x,L*M);

h_pp_size = [L, length(h)/L];
%x_pp_size = [L, length(x)/L];

h_pp = reshape(h, h_pp_size);

row_shift = floor((0:M:L*M-1)/L)+1;
s = zeros(L,length(pp_decimate(x,h_pp(1,:),M)));
for i=1:L
    s(i,:)=pp_decimate(x(1,row_shift(i):end),h_pp(i,:),M);
end

%row_order = mod(0:M:L*M-1,L);
%s = s(row_order+1,:);

y= s(:)';
end

