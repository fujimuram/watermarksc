clear;
Sign = [];
for row = 1:4
    SignTmp = [];
    for col = 1:4
        Place = [row,col];
        WatermarkNumber = (row - 1) * 4 + col;
        WatermarkName = strcat('Watermark/watermark',num2str(WatermarkNumber),'.png');
        SignTmp = [SignTmp rgb2gray(imread(WatermarkName))/ 255];
    end
    Sign = [Sign ; SignTmp];
end
filename = strcat('Watermark/watermark.png');
imwrite(Sign * 255,filename);
imshow(Sign * 255);