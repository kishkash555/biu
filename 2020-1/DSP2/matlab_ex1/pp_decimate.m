function y=pp_decimate(x,h,M)
h = pad_divisible(h,M);
x = pad_divisible(x,M);

h_pp = reshape(h, M, length(h)/M);

t = reshape(-M+1:length(x),M,length(x)/M+1);
t = t(M:-1:1,:);
ind_t = t+1;
ind_t(ind_t<1)=1;
ind_t(ind_t>length(x))=1;
x_pp = x(ind_t);
x_pp(t<0)=0;


y = zeros(1,size(h_pp,2)+size(x_pp,2)-1);

for i=1:M
    curr_x = x_pp(i,:);
    new_comp = conv(h_pp(i,:),curr_x);
    y = y + new_comp;
end

end