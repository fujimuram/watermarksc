function [] = meanAttack(Image,ImageName,a,t)


ImageName = strcat('image/LENNA512.tiff');
ImageName2 = strcat(extractBefore(erase(erase(ImageName,extractBefore(ImageName,'/') ),'/'),'.'));
Image = imread(ImageName);

a = 10;
t = 0.5;


[~, ~, ~] = mkdir(strcat('sign_extract/meanAttack/',ImageName2) );


[EmbedImage,Sign] = makeWatermarkImages(Image,ImageName,a,t);

for i = 1:16
    
    for j = 1:i
        if j == 1
            MeanImage(:,:,i) = double( EmbedImage(:,:,1) );
        else
            MeanImage(:,:,i) = MeanImage(:,:,i) + double( EmbedImage(:,:,j) );
        end
    end
    
    MeanImage(:,:,i) = MeanImage(:,:,i) / i;
end

MeanImage = uint8(MeanImage);


WatermarkName = strcat('Watermark/watermark.png');
SignAll = imread(WatermarkName)/ 255;
for i = 1:16
    
        ExtractSign(:, :, i) = extract_hybrid_DWTnSVD(Image,SignAll,MeanImage(:,:,i),a,t);
        %‰æ‘œ•Û‘¶
        filename = strcat('sign_extract/meanAttack/',ImageName2,'/',ImageName2,'_meanatacked[',num2str(i),']_extract_hybridDWTnSVD','_WM','_kyoudo=',num2str(a),'_threshold=',num2str(t),'.bmp');
        imwrite(ExtractSign(:, :, i),filename);
    
end

%imshow(uint8(MeanImage(:,:,i)));
