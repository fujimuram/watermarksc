

function [result] = emb_hybrid_DWTnSVD(Image,Sign,a)

if size(Image) < size(Sign) / 2 
    disp('サイズが足りません。元画像のサイズは透かし情報の２倍必要です。')
    disp(['元画像 : ',num2str(size(Image)) ,', 透かし情報 : ', num2str(size(Sign))]);
    return;
elseif size(Image) > size(Sign) * 2
    Sign = padarray(Sign,size(Image) / 2 - size(Sign),1,'post'); 
end

%Step 1. 元画像をDWT変換する
[LL, cH, cV, cD] = dwt2(Image,'haar');

%Step 2. LLにSVDを適用
[U_LL, S_LL, V_LL] = svd(LL);


%Step 3. S_LLに透かしを埋め込む
S_LL_D = S_LL + a * double(Sign);

%Step 4. S_LL_DにSVDを適用
[U_SS_L, S_SS_L, V_SS_L] = svd(S_LL_D);

%Step 5. S_SS_LとStep 2.で求めたU_LL,V_LLを用い、透かしが埋め込まれたLL領域(LL_SVD)を復元
LL_SVD = U_LL * S_SS_L * V_LL.';

%Step 6. 透かしが埋め込まれたLL領域(LL_SVD)を使って逆DWTを行う
EmbedImage = uint8( idwt2(LL_SVD,cH,cV,cD,'haar') );


result=EmbedImage;

