gunzip('elvis.bin.gz');

fid = fopen('elvis.bin', 'r');
F = fread(fid, inf, 'double');

I = reshape(F, 700, 469)';
figure; imshow(I);

IMean = mean(I);
IStd = std(I);

[m n] = size(I);

B     = (I-repmat(IMean,[m, 1]))./repmat(IStd,[m, 1]);

C = cov(B);



% A = rot90(I, -1);
% figure; imshow(A)
% 
% Cov_matrix = cov(A);
% 
% Cov_matrix_I = cov(I);


% transpose this to see the normal picture
imshow(reshape(F, 700, 469)');



