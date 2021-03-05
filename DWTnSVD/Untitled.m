clear;

WatermarkNumber = 1;
%ImageName=strcat('image/LENNA512.tiff');
%ImageName=strcat('image/Mandrill512.tiff');
ImageName=strcat('image/peppar512.tiff');
Image = imread(ImageName);


%Sign = padarray(Sign,[64 64],0,'post')/ 255;
a = 5;
t = 0.4;

for a = 5:5:50
    for t = 0:0.1:1
         embANDextract(Image,ImageName,WatermarkNumber,a,t);
    end
end