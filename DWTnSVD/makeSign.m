%ImageSizeで指定したサイズの画像にSignを埋め込めるようにSignを拡張する。元のSignは指定したの場所に入る
%サイズが合わなかったときどうなるかは知らない
function [result] = makeSign(ImageSize,Sign,Place)

%ブロックの個数（行、列）
BlockNumber = fix( (ImageSize / 2) ./ size(Sign) );

TmpSign = padarray(sign(Sign), (Place - [1,1]) .* size(Sign),1,'pre'); 

Sign = padarray(TmpSign, (BlockNumber - Place) .* size(Sign), 1, 'post'); 
%imshow(Sign * 255);

result = Sign;