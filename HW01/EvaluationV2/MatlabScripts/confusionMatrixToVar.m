function [TP FP FN TN stats] = confusionMatrixToVar(confusionMatrix)
    TP = confusionMatrix(1);
    FP = confusionMatrix(2);
    FN = confusionMatrix(3);
    TN = confusionMatrix(4);
    
    if ((TP+FN)==0) %If all GT masks are empty, as in TimeOfDay_ds and Ls_ds (IlluminationChanges)
        recall = 0;
        FNR = 0;
        precision = 0;
        FMeasure = 0;
    else
        recall = TP / (TP + FN);
        FNR = FN / (TP + FN);
        precision = TP / (TP + FP);
        FMeasure = 2.0 * (recall * precision) / (recall + precision);
    end
    specificity = TN / (TN + FP);
    FPR = FP / (FP + TN);
    PWC = 100.0 * (FN + FP) / (TP + FP + FN + TN);
    
    stats = [recall specificity FPR FNR PWC precision FMeasure];
end
