function x_pp = reshape_signal(x,R)

t = reshape(-R+1:length(x),R,length(x)/R+1);
t = t(R:-1:1,:);
ind_t = t+1;
ind_t(ind_t<1)=1;
ind_t(ind_t>length(x))=1;
x_pp = x(ind_t);
x_pp(t<0)=0;

end