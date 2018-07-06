function [ tga ] = compute_blur_metric_spread( I, mask, x, y )
%find the valid point in the mask that nearst to the (i,j)
%   Detailed explanation goes here
    [width, length] = size(mask);
    radius = 8;
    
    mini = max(1, x-radius);
    maxi = min(width, x+radius);
    minj = max(1, y-radius);
    maxj = min(length, y+radius);
    
    dst = 1000;
    val = 0;
%     posx = 0;
%     posy = 0;
    for i = mini:maxi
        for j = minj:maxj
            if mask(i,j) == 0  % find local extreme
                if i == x & j == y
                    continue
                end
                dst_tmp = (i-x)*(i-x)+(j-y)*(j-y);
                if dst_tmp < dst
                    dst = dst_tmp;
                    val = I(i,j);
%                     posx = i;
%                     posy = j;
                end
            end
        end
    end
    if dst == 100
        tga = 0;
    else
        spread = 1.0 * sqrt(dst);
        tga = abs(I(x,y) - val) / spread;
    end
end

