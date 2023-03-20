from imblearn.over_sampling import SMOTE,SVMSMOTE,ADASYN
from imblearn.combine import SMOTETomek,SMOTEENN
from smote_variants import (DBSMOTE,DSMOTE,SMOTE_D,CURE_SMOTE,kmeans_SMOTE,SOMO,NRAS,SYMPROD,G_SMOTE,RWO_sampling,ANS)
from COS_Funcs.baseline.GANs.oversampler import WGAN

def do_oversampling(model,X_train,y_train,*args): 
    
    if model == 'original':
        return X_train,y_train
    
    elif model == 'smote':
        smote = SMOTE()
        return smote.fit_resample(X_train,y_train)
    
    elif model == 'svm_smote':
        svmsmote = SVMSMOTE()
        return svmsmote.fit_resample(X_train,y_train)
        
    elif model == 'smote_enn':
        smoteenn = SMOTEENN()
        return smoteenn.fit_resample(X_train,y_train)
    
    elif model == 'smote_tl':
        smotetl = SMOTETomek()
        return smotetl.fit_resample(X_train,y_train,)
    
    elif model == 'adasyn':
        adasyn = ADASYN()
        return adasyn.fit_resample(X_train,y_train)
    
    elif model == 'db_smote':
        dbsmote = DBSMOTE()
        return dbsmote.sample(X_train,y_train)
    
    elif model == 'd_smote':
        dsmote = DSMOTE()
        return dsmote.sample(X_train,y_train)
    
    elif model == 'smote_d':
        smoted = SMOTE_D()
        return smoted.sample(X_train,y_train)
    
    elif model == 'cure_smote':
        curesmote = CURE_SMOTE()
        return curesmote.sample(X_train,y_train)
    
    elif model == 'kmeans_smote':
        kmeanssmote = kmeans_SMOTE()
        return kmeanssmote.sample(X_train,y_train)
    
    elif model == 'somo':
        somo = SOMO()
        return somo.sample(X_train,y_train)
    
    elif model == 'nras':
        nras = NRAS()
        return nras.sample(X_train,y_train)
    
    elif model == 'symprod':
        symprod = SYMPROD()
        return symprod.sample(X_train,y_train)

    elif model == 'g_smote':
        gsmote = G_SMOTE()
        return gsmote.sample(X_train,y_train)

    elif model == 'rwo_sampling':
        rwo = RWO_sampling()
        return rwo.sample(X_train,y_train)
    
    elif model == 'ans':
        ans = ANS()
        return ans.sample(X_train,y_train)
    
    elif model == 'wgan':
        return WGAN(X_train,y_train)
    
    elif model == 'cos':
        N,c,alpha,linkage,L,shrink_half,expand_half,all_safe_weight,all_safe_gen,half_safe_gen,Gaussian_scale,IR,minlabel,majlabel,visualize = get_cos_para(args[0])
        return cos.COS(X_train,y_train,N,c,alpha,linkage,L,shrink_half,expand_half,all_safe_weight,all_safe_gen,half_safe_gen,Gaussian_scale,IR,minlabel,majlabel,visualize)
    else:
        return 0