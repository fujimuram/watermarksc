function [EmbedImage,Sign] = makeWatermarkImages(Image,ImageName,a,t)

for row = 1:4
    for col = 1:4
        Place = [row,col];
        WatermarkNumber = (row - 1) * 4 + col;
        WatermarkName = strcat('Watermark/watermark',num2str(WatermarkNumber),'.png');
        
        SignTmp = rgb2gray(imread(WatermarkName))/ 255;
        
        Sign(:, :, WatermarkNumber) = makeSign(size(Image),SignTmp,Place);
        
        EmbedImage(:, :, WatermarkNumber) = emb_hybrid_DWTnSVD(Image, Sign(:, :, WatermarkNumber), a);
        filename = strcat('image_embed/',extractBefore(erase(erase(ImageName,extractBefore(ImageName,'/') ),'/'),'.'),'_embed_hybridDWTnSVD','_WM',num2str(WatermarkNumber),'_kyoudo=',num2str(a),'.bmp');
        imwrite( EmbedImage(:, :, WatermarkNumber),filename);
        
        ExtractSign(:, :, WatermarkNumber) = extract_hybrid_DWTnSVD(Image,Sign(:, :, WatermarkNumber),EmbedImage(:, :, WatermarkNumber),a,t);
        
        %‰æ‘œ•Û‘¶
        %filename = strcat('sign_extract/',extractBefore(erase(erase(ImageName,extractBefore(ImageName,'/') ),'/'),'.'),'_embed_hybridDWTnSVD','_WM',num2str(WatermarkNumber),'_kyoudo=',num2str(a),'_threshold=',num2str(t),'.bmp');
        %imwrite(ExtractSign(:, :, WatermarkNumber),filename);
    end
end
