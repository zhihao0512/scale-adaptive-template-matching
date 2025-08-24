function [bWSDIS, brectWSDIS] = computescore(nnf,sT,sI)
%COMPUTESCORE 此处显示有关此函数的摘要
%   此处显示详细说明
pad_nnf=padding(nnf,sT(1:2));
DIS = DIS_scan(int32(pad_nnf),double(sT(1)),double(sT(2)));
mDis = max(DIS(:));
mask=(DIS>mDis*0.7);

bscore=-1;
scale=[1.0 1.0];
for i=0.5:0.1:2
    sS(1)=double(round(sT(1)*i));
    sS(2)=double(round(sT(2)*i));
    if sS(1)>sI(1)||sS(2)>sI(2)
        break;
    end
    WSDIS = WSDIS_scan(int32(nnf),int32(mask),double(sT(1)),double(sT(2)),double(sS(1)),double(sS(2)),30,0.5);
    WSDIS(WSDIS<0)=0;
    % find target
    padMap = padding( ones(size(WSDIS)) , sS(1:2) );
    WSDIS = padding(WSDIS,sS(1:2));

    windowSizeDividor = 3;
    locSearchStyle = 'max';
    centermass = false;
    [rectWSDIS,score]  = findTargetLocation(WSDIS,locSearchStyle,[sS(2) sS(1)], windowSizeDividor, centermass, padMap);

    if score>bscore
        bscore=score;
        brectWSDIS=rectWSDIS;
        bWSDIS=WSDIS;
        scale=[i,i];
    end
end

