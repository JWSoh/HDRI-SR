clear;
close all;

Set={'Test','Part1','Part2','Part3'};

for num=1:length(Set)
    setname=Set{num};
    Path= sprintf('TestImg/%s/',setname);
    Path2 =  sprintf('REF_result/%s/',setname);
    
    images = dir([Path, '*.png']);
    reflect = dir([Path2, '*.mat']);
    
    gamma = 2.2;
    
    scale=2;
    
    D_max=1;
    D_min=0;
    
    if ~exist(sprintf('HDR_result/%s',setname),'file')
        mkdir(sprintf('HDR_result/%s',setname));
    end
    
    for idx = 1:length(images)
        images(idx).name
        
        img= im2double(imread([Path images(idx).name]));
        img_yuv= rgb2ycbcr(img);
        
        REF = load([Path2, reflect(idx).name]);
        REF= REF.REF;
        REF=min(max(REF,-1),1);
        REF = atanh(REF);
                
        [hei,wid]=size(REF);
  
        L=img_yuv(:,:,1);
        
        ILL = wlsFilter(L.^2.2, 2, 2);
        ILL = imresize(ILL,[hei,wid],'bicubic');
        ILL = max(ILL,0);
        
        EN_LUM = exp(REF) .* ILL.^(1/2.2);
        
        RES=cat(3, EN_LUM, imresize(img_yuv(:,:,2),[hei,wid],'bicubic'),imresize(img_yuv(:,:,3),[hei,wid],'bicubic'));
        RES=double(RES);
        
        RES=ycbcr2rgb(RES);
        RES=RES.^gamma;
        RES_hdr=RES*(D_max-D_min)+D_min;
        hdrwrite(RES,sprintf('HDR_result/%s/%s.hdr',setname,images(idx).name(1:end-4)));

    end
end