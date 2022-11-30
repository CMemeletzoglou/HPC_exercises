% File       : pca_demo.m
% Description: Demo for principal component analysis with the king.  Visit also:
%              http://matlabdatamining.blogspot.ch/2010/02/putting-pca-to-work.html
clear('all');

keepN = 50;

% read the image
I = imread('elvis.jpg');
A = double(rgb2gray(I));

% first stage: construction of the covariance matrix
[m, n] = size(A);
AMean = mean(A); % row vector, where the i-th element is the mean of the i-th column
AStd  = std(A); % row vector, where the i-th element is the std dev of the i-th column

% the first part, substracts from each image column its mean value
% normalize each image column with **its** mean value and **its** std dev
B     = (A-repmat(AMean,[m, 1]))./repmat(AStd,[m, 1]);

C     = cov(B);

% second stage: computation of eigenvalues/eigenvectors
[V,D] = eig(C);

% third stage: principal components
if keepN > n
    keepN = n;
end
VReduced  = V(:, (n-keepN+1):n);
PCReduced = B*VReduced;

% reconstruct compressed image
Z = ((PCReduced * VReduced') .* repmat(AStd,[m, 1])) + repmat(AMean,[m, 1]);

% show the image
figure;imshow(uint8(round(Z)))
