% File       : show_image.m
% Description: Show gray-scale image from binary file
clear('all');

filename = 'elvis.50.bin'

A = load(filename);
imshow(A);
