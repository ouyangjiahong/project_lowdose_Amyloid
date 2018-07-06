function [ FM ] = blur_metric_freq( I )
% metric for blurred image in frequency domain
%   Image Sharpness Measure for Blurred Images in Frequency Domain

    % Input: I of M x N
    % Output: Image Quality measure (FM) where FM stands for Frequency Domain Image Blur Measure
    [width, length] = size(I);

    % step1: Compute F which is the Fourier Transform representation of image I
    F = fft2(I);

    % step2: Find Fc which is obtained by shifting the origin of F to centre.
    Fc = fftshift(F);

    % step3: AF is the absolute value of the centered Fourier transform of image I
    AF = abs(Fc);

    % step4: M is the maximum value of the frequency component in F.
    M = max(AF(:));

    % step 5: Calculate T = the total number of pixels in F whose pixel value > thres,where thres = M/1000.
    thres = M / 1000.0;
    tmp = abs(F) > thres;
    T = sum(tmp(:));

    % step6: Calculate Image Quality measure (FM)from: image quality measure FM=T/(MN)
    FM = T / (1.0 * length * width);


end

