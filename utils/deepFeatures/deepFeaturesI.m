function Fmap = deepFeaturesI(net,im,gpuN)
% note: im in 255 range !!
%indLayers = [37, 28, 19, 10, 5];   % The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
im_ = preProcessing(im, net.meta.normalization.averageImage,gpuN);
[w,h,~] = size(im);

padsize = 16;
im_pad = padarray(im_,[padsize padsize],'symmetric');
[w_pad,h_pad,~] = size(im_pad);

res = vl_simplenn(net, im_pad) ;

% pz = 3;
% Cmap = zeros(w,h,pz*pz);

% conv1_1 = gather(res(3).x);
% conv1_1_re = imresize(conv1_1, [w,h]);

conv1_2 = gather(res(5).x);
conv1_2_re_pad = imresize(conv1_2, [w_pad,h_pad]);
l2_1 = repmat(sqrt(max(sum(conv1_2_re_pad.^2,3),1e-12)),1,1,size(conv1_2_re_pad,3));
conv1_2_re_pad = conv1_2_re_pad./l2_1;
conv1_2_re = conv1_2_re_pad(padsize+1:padsize+w,padsize+1:padsize+h,:);

% conv2_1 = gather(res(8).x);
% conv2_1_re = imresize(conv2_1, [w,h]);

% conv2_2 = gather(res(10).x);
% conv2_2_re = imresize(conv2_2, [w,h]);

conv3_4 = gather(res(19).x);
conv3_4_re_pad = imresize(conv3_4, [w_pad,h_pad]);
l2_2 = repmat(sqrt(max(sum(conv3_4_re_pad.^2,3),1e-12)),1,1,size(conv3_4_re_pad,3));
conv3_4_re_pad = conv3_4_re_pad./l2_2;
conv3_4_re = conv3_4_re_pad(padsize+1:padsize+w,padsize+1:padsize+h,:);

% conv4_4 = gather(res(28).x);
% conv4_4_re_pad = imresize(conv4_4, [w_pad,h_pad]);
% conv4_4_re = conv4_4_re_pad(padsize+1:padsize+w,padsize+1:padsize+h,:);


Fmap = cat(3,conv1_2_re,conv3_4_re);
end


function im_out = preProcessing(im_in,averageImage,gpuN)
im_in = single(im_in); 
%im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_out = bsxfun(@minus,im_in,averageImage) ;
if gpuN>0
    im_out = gpuArray(im_out);
end
end




