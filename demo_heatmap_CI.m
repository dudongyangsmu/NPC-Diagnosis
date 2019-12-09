subfile = {'Fea_IBSI_tr_te_t5_CI'}; 
for j = 1:length(subfile)
    DIR = dir([pwd,'\','results','\',subfile{j},'\']); 
    CaseName = struct2cell(DIR);
    for i  = 1:size(CaseName,2)-2                     
        classification_method= CaseName{1,i+2};
        selection_method={'fisher','relieff','mrmr','mim','cmim','jmi'};
        for k=1:length(selection_method)
            load([pwd,'\','results','\',subfile{j},'\',classification_method,'\',selection_method{k},classification_method,'.mat']);
          
            auc(k,i)=stats.auc(1);
            spe(k,i)=stats.spe(1);
            sen(k,i)=stats.sen(1);
            ACC(k,i)=stats.ACC(1);    
            
            auc_low(k,i)=stats.auc(2);
            spe_low(k,i)=stats.spe(2);
            sen_low(k,i)=stats.sen(2);
            ACC_low(k,i)=stats.ACC(2); 
            
            auc_up(k,i)=stats.auc(3);
            spe_up(k,i)=stats.spe(3);
            sen_up(k,i)=stats.sen(3);
            ACC_up(k,i)=stats.ACC(3); 
            
        end
        error=1-ACC;
        error_low=1-ACC_up;
        error_up=1-ACC_low;
    end
    
    save auc auc
    save spe spe
    save sen sen
    save ACC ACC
    save error error
    
     save auc_low auc_low
    save spe_low spe_low
    save sen_low sen_low
    save ACC_low ACC_low
    save error_low error_low
    
    save auc_up auc_up
    save spe_up spe_up
    save sen_up sen_up
    save ACC_up ACC_up
    save error_up error_up
    
end