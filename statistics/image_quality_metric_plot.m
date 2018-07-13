figure(1)
method = {'low-dose PET';'U-Net';'GAN';'2.5D GAN'};
% PSNR
mean = [22.4, 27.7, 27.6, 28.5]; 
std = [15.96, 10.35, 9.76, 9.5]; 
subplot(1,3,1)
hold on
bar(mean,0.8,'FaceColor',[0.75,0.75,0.75])
errorbar(mean, std, '.','Color',[0,0,0])
set(gca, 'XTick',1:4, 'XTickLabel',method, 'TickLabelInterpreter','none',...
    'FontSize',5.5,'YLim', [5 40])
title('PSNR')

% SSIM
mean = [0.86,0.93,0.93,0.945]; 
std = [0.008, 0.0028,0.0030,0.0024]; 
subplot(1,3,2)
hold on
bar(mean,0.8,'FaceColor',[0.75,0.75,0.75])
errorbar(mean, std, '.','Color',[0,0,0])
set(gca, 'XTick',1:4, 'XTickLabel',method, 'TickLabelInterpreter','none',...
    'FontSize',5.5,'YLim', [0.84 0.95])
title('SSIM')

% RMSE
mean = [0.55,0.34,0.35,0.28]; 
std = [0.052,0.056,0.060,0.04]; 
subplot(1,3,3)
hold on
bar(mean,0.8,'FaceColor',[0.75,0.75,0.75])
errorbar(mean, std, '.','Color',[0,0,0])
set(gca, 'XTick',1:4, 'XTickLabel',method, 'TickLabelInterpreter','none',...
    'FontSize',5.5,'YLim', [0.25 0.65])
title('RMSE')

figure(2)
method = {'standard-dose PET';'U-Net';'GAN';'2.5D GAN'};
% FBM
mean = [0.086,0.066,0.085,0.08]; 
std = [0.0072,0.0034,0.0068,0.0087]; 
subplot(1,3,1)
hold on
bar(mean,0.8,'FaceColor',[0.75,0.75,0.75])
errorbar(mean, std, '.','Color',[0,0,0])
set(gca, 'XTick',1:4, 'XTickLabel',method, 'TickLabelInterpreter','none',...
    'FontSize',5.5,'YLim', [0.06 0.095])
title('FBM')

% EBM
mean = [0.0169,0.0158,0.0164,0.0165]; 
std = [2.0311e-05,3.5869e-05,2.2957e-05,2.8157e-05]; 
subplot(1,3,2)
hold on
bar(mean,0.8,'FaceColor',[0.75,0.75,0.75])
errorbar(mean, std, '.','Color',[0,0,0])
set(gca, 'XTick',1:4, 'XTickLabel',method, 'TickLabelInterpreter','none',...
    'FontSize',5.5,'YLim', [0.015 0.017])
title('EBM')

% subplot(1,3,3)
% hold on
% bar(mean,0.8,'FaceColor',[0.75,0.75,0.75])
% errorbar(mean, std, '.','Color',[0,0,0])
% set(gca, 'XTick',1:3, 'XTickLabel',method, 'TickLabelInterpreter','none',...
%     'FontSize',5.5,'YLim', [0.015 0.017])
% title('EBM')


