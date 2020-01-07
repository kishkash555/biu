clear all; close all; clc;
 
n=4;
m=4;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate MRF model
 
x=randi([0 1],m,n);
MaxSamples=100000;
figure(1);
image([0.5 4.5], [0.5 4.5], 100*x);
colormap(gray(2));
a = axis;
axis([a(1) a(2) 0 4]);
set(gca, 'YTick', 0:4)
set(gca, 'XTick', 0:4)
grid;
for t=1:MaxSamples
  for i=1:m
    for j=1:n
        s1=0; 
        neib=(i>1)+(i<m)+(j>1)+(j<n);
        if (i>1);  s1=s1+x(i-1,j);end;
        if (i<m); s1=s1+x(i+1,j); end;
        if (j>1);  s1=s1+x(i,j-1); end;
        if (j<n); s1=s1+x(i,j+1); end;
        s0=neib-s1;
        p0=exp(s0);
        p1=exp(s1);
        p0=p0/(p0+p1);
        p1=1-p0;
        uni=rand(1);
        if (uni<p0)
            x(i,j)=0;
        else
            x(i,j)=1;
        end  
    end;
   end;
end;
 
figure(2);
image([0.5 4.5], [0.5 4.5], 100*x);
colormap(gray(2));
a = axis;
axis([a(1) a(2) 0 4]);
set(gca, 'YTick', 0:4)
set(gca, 'XTick', 0:4)
grid;
 
y=x+randn(m,n);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute the marginal
 
MaxSeq=2^(m*n);
px_y=zeros(m,n,2);
 
h = waitbar(0, 'Please wait');
for k=0:MaxSeq-1
    waitbar(k/MaxSeq, h);
    vec=zeros(1,m*n);
    %ascii of '1' is 49
    seq=dec2bin(k)-48;
    %binary2vector
    vec(1:length(seq))=seq(end:-1:1);
    z=reshape(vec,n,m)';      
    sum1=0;
    for i=1:m-1
        for j=1:n-1
                sum1=sum1+(z(i,j)==z(i+1,j))+(z(i,j)==z(i,j+1));
        end
    end 
 
      %last row
      for j=1:n-1; sum1=sum1+(z(m,j)==z(m,j+1)); end;
      %last col
      for i=1:m-1; sum1=sum1+(z(i,n)==z(i+1,n)); end;
 
        sum2=sum(sum(-0.5*(z-y).^2));
         pz_y=exp(sum1+sum2);
        for i=1:m
            for j=1:n
                 px_y(i,j,z(i,j)+1)=px_y(i,j,z(i,j)+1)+pz_y;
            end
        end
end
close(h);
tot=sum(px_y,3);
px_y(:,:,1)=px_y(:,:,1)./tot;
px_y(:,:,2)=px_y(:,:,2)./tot;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% apply Gibbs sampling to compute  marginal
  
g=randi([0 1],m,n);
count=zeros(m,n);
err=zeros(1,MaxSamples);
 
for t=1:MaxSamples
    for i=1:m
        for j=1:n
            s1=0;
            neib=(i>1)+(i<m)+(j>1)+(j<n);
            if (i>1);  s1=s1+g(i-1,j);end;
            if (i<m); s1=s1+g(i+1,j); end;
            if (j>1);  s1=s1+g(i,j-1); end;
            if (j<n); s1=s1+g(i,j+1); end;
            s0=neib-s1;
            s0=s0-0.5*(0-y(i,j))^2;
            s1=s1-0.5*(1-y(i,j))^2;
            p0_y=exp(s0);
            p1_y=exp(s1);
            p0_y=p0_y/(p0_y+p1_y);
            p1_y=1-p0_y;
            uni=rand(1);
            if (uni<p0_y)
                g(i,j)=0;
            else
                g(i,j)=1;
            end  
            count(i,j)=count(i,j)+g(i,j);
        end;
    end;
    p_est=zeros(m,n,2);
    p_est(:,:,2)=count/t;
    p_est(:,:,1)=(t-count)/t;
    err(t)=sum(sum((p_est(:,:,1)-px_y(:,:,1)).^2));
end;
 
figure(3);
semilogy(err)
