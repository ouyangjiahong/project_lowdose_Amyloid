function [ tga_mean ] = blur_metric_time( I )
% metric for blurring in time domain
%   Blind Image Quality Assessment for Measuring Image Blur
    
    [width, length] = size(I);
    sobel = fspecial('sobel');
    sigma = 0;
    
    % step1: canny edge detection
    edges = edge(I, 'canny');

    % step2: gradient direction detection
    [mag, dir] = imgradient(I, 'sobel'); 
    dir(edges==0) = 0;

    % step3: measure edge-spread
    tga_sum = 0;
    tga_num = 0;
    for i = 1:width
        for j = 1:length
            if dir(i,j) ~= 0
                % the filter in direction dir(i,j)
                filter = sobel * cos(dir(i,j));
                % the gradient in direction dir(i,j)
                grad_dir = imfilter(I, filter); 
                grad_dir(abs(grad_dir)>sigma)=1;
                tga = compute_blur_metric_spread(I, grad_dir, i, j);
                if tga ~= 0
                    tga_sum = tga + tga_sum;
                    tga_num = tga_num + 1;
                end
            end
        end
    end
    tga_mean = tga_sum / (1.0*tga_num);

end

