function Fmap = deepFeaturesT(net,im,gpuN)
% note: im in 255 range !!
%indLayers = [37, 28, 19, 10, 5];   % The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
im_ = preProcessing(im, net.meta.normalization.averageImage,gpuN);
[w,h,~] = size(im);

res = vl_simplenn(net, im_) ;

conv1_2 = gather(res(5).x);
l2_1 = repmat(sqrt(max(sum(conv1_2.^2,3),1e-12)),1,1,size(conv1_2,3));
conv1_2 = conv1_2./l2_1;
conv1_2_re = imresize(conv1_2, [w,h]);

% conv2_2 = gather(res(10).x);
% conv2_2_re = imresize(conv2_2, [w,h]);

conv3_4 = gather(res(19).x);
l2_2 = repmat(sqrt(max(sum(conv3_4.^2,3),1e-12)),1,1,size(conv3_4,3));
conv3_4 = conv3_4./l2_2;
conv3_4_re = imresize(conv3_4, [w,h]);

% conv4_4 = gather(res(28).x);
% conv4_4_re = imresize(conv4_4, [w,h]);

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




