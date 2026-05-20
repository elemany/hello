function [acc, F1_raw] = main(subjectIndex, nb_senator)
if nargin<1, subjectIndex=1; end
if nargin<2, nb_senator=3; end
rng(7);
uS=6; uN=700; uD=40;
Wrest=cell(uS,1); Wtask=cell(uS,1); Wlab=cell(uS,1);
for uu=1:uS
    m1=(uu-1)*0.2; m2=1.2+(uu-1)*0.2;
    A1=mvnrnd(m1*ones(1,uD),0.5*eye(uD),round(uN/2));
    A2=mvnrnd(m2*ones(1,uD),0.5*eye(uD),uN-round(uN/2));
    B1=mvnrnd((m1+0.5)*ones(1,uD),0.6*eye(uD),round(uN/2));
    B2=mvnrnd((m2+0.5)*ones(1,uD),0.6*eye(uD),uN-round(uN/2));
    Wrest{uu}=A1(randperm(size(A1,1),min(300,size(A1,1))),:);
    Wrest{uu}=[Wrest{uu};A2(randperm(size(A2,1),min(300,size(A2,1))),:)];
    Wtask{uu}=[B1;B2];
    Wlab{uu}=[ones(size(B1,1),1);2*ones(size(B2,1),1)];
end
tgt=subjectIndex;
if tgt<1||tgt>uS, tgt=1; end
dL=zeros(uS,1);
for jj=1:uS
    if jj==tgt
        dL(jj)=inf; 
    else
        dL(jj)=KLD(Wrest{tgt},Wrest{jj},3);
    end
end
[~,pS]=sort(dL);
pS=pS(1:nb_senator);
qX=cell(numel(pS),1); qY=cell(numel(pS),1);
for kk=1:numel(pS)
    qX{kk}=Wtask{pS(kk)};
    qY{kk}=Wlab{pS(kk)};
end
zA=Wrest{tgt};
zB=Wtask{tgt};
zY=Wlab{tgt};
gX=[];
for kk=1:numel(pS)
    gX=[gX;qX{kk}];
end
gX=[gX;zA];
vDim=12;
R=PCA1(gX,vDim);
gX2=gX*R;
sCnt=0; sPos=0;
X_src=cell(numel(pS),1);
for kk=1:numel(pS)
    sPos=sCnt+1; sCnt=sCnt+size(qX{kk},1);
    X_src{kk}=gX2(sPos:sCnt,:);
end
X_tar=gX2(sCnt+1:end,:);
opt2.beta=1; opt2.gamma=1; opt2.g1=1; opt2.T=1; opt2.src_n=numel(pS); opt2.k=min(6,vDim);
[~, Pc, P]=ESPA(X_src, qY, X_tar, opt2);
Ttr=[]; Tl=[];
for kk=1:numel(pS)
    Ttr=[Ttr; (P{kk}*X_src{kk}')'];
    Tl=[Tl; qY{kk}];
end
Zt=(Pc*(zB*R)')';
k0=5; mfz=2;
pred1=nihao(Ttr,Tl,Zt,k0,mfz);

% [fis_trn, trErr, stepSize] = anfis([Zs', Zs_label], opt); 
% L_B_predicted_LVQ_train = round(evalfis(fis_trn, Zt'));  
% L_B_predicted_LVQ_train(L_B_predicted_LVQ_train < 1) = 1;
% L_B_predicted_LVQ_train(L_B_predicted_LVQ_train > 2) = 2;
% acc_supervised = length(find(L_B_predicted_LVQ_train==L_B'))/length(L_B); 
% acc_transfered_LVQ = acc_supervised;  % 保存每轮结果  
% pred_label1 = L_B_predicted_LVQ_train;  

acc=sum(pred1==zY)/numel(zY);
F1_raw=F1macro(zY,pred1);

end