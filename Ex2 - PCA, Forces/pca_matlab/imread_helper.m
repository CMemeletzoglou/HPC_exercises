gunzip('elvis.bin.gz');

fid = fopen('elvis.bin', 'r');
F = fread(fid, inf, 'double');

I = reshape(F, 700, 469)';
figure; imshow(I);


A = rot90(I, -1);
figure; imshow(A)







% transpose this to see the normal picture
imshow(reshape(F, 700, 469)');


