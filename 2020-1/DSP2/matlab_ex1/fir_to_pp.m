function pp_filter = fir_to_pp(L,M, orig_filter)
    %FIR_TO_PP get the filter coefficients in polyphase format
    Lh = length(orig_filter);
    if mod(Lh,L*M) ~= 0
        padlen = L-mod(Lh,L*M);
        orig_filter = padarray(orig_filter, [0, padlen],'post');
        Lh = length(orig_filter);
    end
   
    pp_filter = reshape(orig_filter,[L*M ,Lh/(L*M)]);
    
end

