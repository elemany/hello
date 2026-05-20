function [obj,Pc,P,alpha] = ESPA(X_src,X_src_label,X_tar_train,options)

tauMax   = options.T;
nSoup    = options.src_n;
kk       = options.k;
bta      = options.beta;
gm       = options.gamma;
gee1     = options.g1;

cfg2 = [];
cfg2.T      = 1500;
cfg2.beta   = bta;
cfg2.gamma  = gm;
cfg2.g1     = gee1;
cfg2.src_n  = nSoup;
cfg2.k      = kk;

for loopTick = 1:tauMax
    [obj,P,Pc,alpha] = myESPA(X_src,X_src_label,X_tar_train,cfg2);
end
end


function [obj,P,Pc,alpha] = myESPA(X_src,X_src_label,X_tar_train,options)

tauInner = options.T;
nSoup    = options.src_n;
kk       = options.k;
bta      = options.beta;
gm       = options.gamma;
gee1     = options.g1;


bagS = [];          
tagS = [];        
panT = X_tar_train; 
featDim = size(X_src{1},2);


P = cell(1,nSoup);
dimCfg = [];
dimCfg.ReducedDim = kk;
for srcIdx = 1:nSoup
    P{srcIdx} = PCA1(X_src{srcIdx}, dimCfg);
    P{srcIdx} = P{srcIdx}';
end


mapList = cell(1,nSoup);
for srcIdx = 1:nSoup
    mapList{srcIdx} = ones(size(X_src{srcIdx},1), size(X_tar_train,1));
end


tiny = 1e-5;
WinScatter = cell(1,nSoup);
BetScatter = cell(1,nSoup);
Lmix       = cell(1,nSoup);

for srcIdx = 1:nSoup
    [WinScatter{srcIdx}, BetScatter{srcIdx}] = ScatterMat(X_src{srcIdx}', X_src_label{srcIdx});
    Lmix{srcIdx} = WinScatter{srcIdx} - tiny * BetScatter{srcIdx};
    bagS = [bagS; X_src{srcIdx}];
    tagS = [tagS; X_src_label{srcIdx}];
end

stackAll = [bagS; panT];
Pc = PCA1(stackAll, dimCfg);
Pc = Pc';


alpha = ones(nSoup,1) / nSoup;
alpha_accum = alpha;


obj = zeros(1, tauInner);
for roundK = 1:tauInner
    for ss = 1:nSoup
        numerP = alpha(ss)*gee1*Pc*panT'*mapList{ss}'*X_src{ss} + bta*Pc;
        denomP = alpha(ss)*Lmix{ss} + alpha(ss)*gee1*X_src{ss}'*mapList{ss}*mapList{ss}'*X_src{ss} + bta*eye(featDim);
        P{ss}  = numerP / denomP;
        ridgeTau = 1e-4;
        Ablk = X_src{ss} * P{ss}' * P{ss} * X_src{ss}' + ridgeTau * eye(size(X_src{ss},1));
        Bblk = X_src{ss} * P{ss}' * Pc * panT';
        mapList{ss} = Ablk \ Bblk;
        numerC = alpha(ss)*gee1*P{ss}*X_src{ss}'*mapList{ss}*panT + bta*P{ss};
        denomC = alpha(ss)*gee1*(panT'*panT) + bta*eye(featDim);
        Pc     = numerC / denomC;


        alpha_accum(ss) = trace(P{ss}*Lmix{ss}*P{ss}') + ...
                          gee1 * norm(P{ss}*X_src{ss}'*mapList{ss} - Pc*panT', 'fro');
    end

    vscore = alpha_accum / (2 * gm);
    vscore = normalization(vscore', 3);
    vscore = vscore';
    alpha  = EProjSimplex_new(vscore, 1);

    bucket = 0;
    for ss = 1:nSoup
        bucket = bucket ...
               + alpha(ss)*trace(P{ss}*Lmix{ss}*P{ss}') ...
               + alpha(ss)*gee1*norm(P{ss}*X_src{ss}'*mapList{ss} - Pc*panT', 'fro') ...
               + bta*norm(P{ss} - Pc, 'fro');
    end
    obj(roundK) = bucket + gm * sum(alpha.^2);

    if (roundK >= 2) && (abs(obj(roundK) - obj(roundK-1)) < 1)
        disp(['MDSA iter ' num2str(roundK) ' convergence.............., the obj_value is ' num2str(obj(roundK))]);
        break;
    end
end
end
