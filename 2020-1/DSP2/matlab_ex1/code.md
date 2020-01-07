## Resampling code
The code works for both L > M and M L < M. For very high decimation rate, there is not enough bandwidth to carry the speech signal and some information is lost.

### resample.m - main function
```matlab
function y = resample(x,L,M)
    %RESAMPLE Summary of this function goes here
    %   Detailed explanation goes here
    h = create_lpf(L,M);
    g = fir_to_pp(L,M,h);
    y = run_pp_filter(x,g,L,M);
end
```

### create_lpf.m
```matlab
function fir_coeffs = create_lpf(L,M)
    %CREATE_LPF create an FIR LPF for rational rate change
    wc = min(1/L,1/M);
    fir_coeffs = L*fir1(200,wc,'low');
end
```

### fir_to_pp.m
```matlab
function pp_filter = fir_to_pp(L,M, orig_filter)
    %FIR_TO_PP get the filter coefficients in polyphase format
    Lh = length(orig_filter);
    if mod(Lh,L) ~= 0
        padlen = L-mod(Lh,L);
        orig_filter = padarray(orig_filter, [0, padlen],'post');
        Lh = length(orig_filter);
    end
   
    pp_filter = reshape(orig_filter,[L ,Lh/L]);
    
end
```

### run_pp_filter.m

```matlab
function y = run_pp_filter(x, pp_filter, L, M)
    %RUN_PP_FILTER perform the filtered resampling using the polyphase filter
    [r,Q] = size(pp_filter);
    len_out = ceil(length(x)*L/M);
    x = padarray(x,[ceil(Q/2), 0]);
    y = zeros(len_out,1);
    for n = 0:len_out-1
        curr_x = floor(n*M/L);
        curr_buffer = x(curr_x+1: curr_x+Q);
        curr_filter = pp_filter(mod(n*M,L)+1,:);
        v = dot(curr_filter, curr_buffer );
        y(n+1,1) = v;
    end
end
```


