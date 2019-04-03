subfile = {'Fea_IBSI_tr_te_t5'}; 
for j = 1:length(subfile)
    DIR = dir([pwd,'\','results','\',subfile{j},'\']);
    CaseName = struct2cell(DIR);
    for i  = 1:size(CaseName,2)-2                     
        classification_method= CaseName{1,i+2};
        selection_method={'fisher','relieff','mrmr','mim','cmim','jmi'};
        for k=1:length(selection_method)
            load([pwd,'\','results','\',subfile{j},'\',classification_method,'\',selection_method{k},classification_method,'.mat']);
            auc(k,i)=stats.auc;
            acc(k,i)=stats.acc;
            testerror=1-acc;
            spe(k,i)=stats.spe;
            sen(k,i)=stats.sen;
        end
    end
    save auc auc
    save testerror testerror
    save spe spe
    save sen sen
end