clear
% load data
l1_dir = '../test/petonly_l1only/';
gan_dir = '../test_25D/petonly_gan+l1/';

FM_list_l1 = [];
FM_list_gan = [];
FM_list_gt = [];

for i = 1 : 162
    for j = 0 : 3
        img_path = ['test_',num2str(i,'%04d'),'_',num2str(j)];
        img_path_l1 = [l1_dir,img_path,'_output.jpg'];
        img_l1 = imread(img_path_l1);
        img_l1 = im2double(img_l1);
        FM_l1 = blur_metric_freq(img_l1);
        FM_list_l1 = [FM_list_l1 FM_l1];
        
        img_path_gan = [gan_dir,img_path,'_output.jpg'];
        img_gan = imread(img_path_gan);
        img_gan = im2double(img_gan);
        FM_gan = blur_metric_freq(img_gan);
        FM_list_gan = [FM_list_gan FM_gan];
        
        img_path_gt = [gan_dir,img_path,'_target.jpg'];
        img_gt = imread(img_path_gt);
        img_gt = im2double(img_gt);
        FM_gt = blur_metric_freq(img_gt);
        FM_list_gt = [FM_list_gt FM_gt];
        
%         F_l1 = fftshift(fft2(img_l1));
%         F_gan = fftshift(fft2(img_gan));
%         F_gt = fftshift(fft2(img_gt));
%         figure(1)
%         subplot(2,3,1);
%         imshow(img_gt);
%         subplot(2,3,2);
%         imshow(img_gan);
%         subplot(2,3,3);
%         imshow(img_l1);
%         subplot(2,3,4);
%         imshow(F_gt);
%         subplot(2,3,5);
%         imshow(F_gan);
%         subplot(2,3,6);
%         imshow(F_l1);
%         
%         a = 1
    end
end

FM_l1_mean = mean(FM_list_l1);
FM_l1_var = var(FM_list_l1);
FM_gan_mean = mean(FM_list_gan);
FM_gan_var = var(FM_list_gan);
FM_gt_mean = mean(FM_list_gt);
FM_gt_var = var(FM_list_gt);
        