function [result] = extract_hybrid_DWTnSVD(Image,Sign,EmbedImage,a,t)


if size(Image) < size(Sign) / 2 
    disp('サイズが足りません。元画像のサイズは透かし情報の２倍必要です。')
    disp(['元画像 : ',num2str(size(Image)) ,', 透かし情報 : ', num2str(size(Sign))]);
    return;
elseif size(Image) > size(Sign) * 2
    Sign = padarray(Sign,size(Image) / 2 - size(Sign),1,'post');
end

% 埋め込み手順のStep1からStep4を実行
[cA, cH, cV, cD] = dwt2(Image,'haar');
[U_LL, S_LL, V_LL] = svd(cA);
S_LL_D = S_LL + a * double(Sign);
[U_SS_L, S_SS_L, V_SS_L] = svd(S_LL_D);



%Step 1. 透かし入り画像にDWT
[LL_W,cH_W,cV_W,cD_W] = dwt2(EmbedImage,'haar');

%Step 2. "透かし入り画像をDWTして得られたLL領域LL_W"にSVDを適用
[U_W,S_W,V_W] = svd(LL_W);

%Step3. 埋め込み手順Step4で求めたU_SS_LとV_SS_Lと抽出手順Step2で求めたS_Wを使って、再構成されたLL領域Construct_S_LLDを得る
Construct_S_LLD = U_SS_L * S_W * V_SS_L.';
%Construct_S_LLD = LL_W;

%Step4. 透かし画像を抽出するための準備。
Construct_W = (Construct_S_LLD - S_LL) / a;

%Step5. 透かしを抽出する。閾値で透かしの有り無しを判定する
Construct_W (Construct_W > t) = 255;
Construct_W (Construct_W <= t) = 0;

result = Construct_W;

