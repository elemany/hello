function F1_new = F1macro(L_B, pred_label1);
confMat = confusionmat(L_B', pred_label1);

    if size(confMat,1) == 2  
        TP = confMat(2,2);

        FP = confMat(1,2);

        FN = confMat(2,1);

        precision_LVQ = TP / (TP + FP + eps);

        recall_LVQ    = TP / (TP + FN + eps);

        F1_LVQ = 2 * precision_LVQ * recall_LVQ / (precision_LVQ + recall_LVQ + eps);

    else  
        numClass = size(confMat, 1);

        F1_each = zeros(numClass, 1);

        for k = 1:numClass
            TP = confMat(k,k);

            FP = sum(confMat(:,k)) - TP;

            FN = sum(confMat(k,:)) - TP;

            precision_k = TP / (TP + FP + eps);

            recall_k    = TP / (TP + FN + eps);

            F1_each(k)  = 2 * precision_k * recall_k / (precision_k + recall_k + eps);

        end

        F1_LVQ = mean(F1_each); 

    end

    F1_new = F1_LVQ;
end