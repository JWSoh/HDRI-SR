clear;
close all;

Set={'Part1','Part2','Part3','Test'};
    
for id=1:length(Set)
    Path=Set{id};
    imgType = '*.png';
    
    images  = dir(fullfile('TestImg',Path,imgType));
    
    scale=2;
    
    mat_path=sprintf('LR_mat(x%d)/%s', scale, Path);
    for idx = 1:length(images)
        if ~exist(mat_path,'dir')
            mkdir(mat_path);
        end
        images(idx).name;
        img = im2double(imread(fullfile('TestImg',Path,images(idx).name)));
        
        L= rgb2ycbcr(img);
        L= L(:,:,1);
        L=L.^2.2;
        ILL = wlsFilter((L),2,2);
        REF = log(L*255+0.0001)-log(ILL*255+0.0001);
       
        
        save(sprintf('LR_mat(x%d)/%s/%s.mat', scale, Path, images(idx).name(1:end-4)),'ILL','REF');
        
    end
end