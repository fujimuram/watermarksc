

function [] = embANDextract(Image,ImageName,WatermarkNumber,a,t)

WatermarkName=strcat('Watermark/watermark',num2str(1),'.png');
Sign = rgb2gray(imread(WatermarkName))/ 255;

ImageName2 = strcat(extractBefore(erase(erase(ImageName,extractBefore(ImageName,'/') ),'/'),'.'));
[~, ~, ~] = mkdir(strcat('sign_extract/',ImageName2) );

%ñÑÇﬂçûÇ›
EmbedImage = emb_hybrid_DWTnSVD(Image,Sign,a);

%âÊëúï€ë∂
filename = strcat('image_embed/',extractBefore(erase(erase(ImageName,extractBefore(ImageName,'/') ),'/'),'.'),'_embed_hybridDWTnSVD','_WM',num2str(WatermarkNumber),'_kyoudo=',num2str(a),'.bmp');
imwrite(EmbedImage,filename);

%íäèo
ExtractSign = extract_hybrid_DWTnSVD(Image,Sign,EmbedImage,a,t);
ExtractSignNotembed = extract_hybrid_DWTnSVD(Image,Sign,Image,a,t);
%âÊëúï€ë∂
filename = strcat('sign_extract/',ImageName2,'/',ImageName2,'_embed_','_WM',num2str(WatermarkNumber),'_kyoudo=',num2str(a),'_thres=',num2str(t),'.bmp');
imwrite(ExtractSign,filename);
filename = strcat('sign_extract/',ImageName2,'/',ImageName2,'_notembed_','_WM',num2str(WatermarkNumber),'_kyoudo=',num2str(a),'_thres=',num2str(t),'.bmp');
imwrite(ExtractSignNotembed,filename);
%subplot(1,2,1),imshow(ExtractSign * 255);
%subplot(1,2,2),imshow(ExtractSignNotembed * 255);