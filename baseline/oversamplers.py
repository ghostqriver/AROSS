from imblearn.over_sampling import SMOTE,SVMSMOTE,ADASYN,RandomOverSampler
from imblearn.combine import SMOTETomek,SMOTEENN
from smote_variants import (DBSMOTE,DSMOTE,SMOTE_D,CURE_SMOTE,kmeans_SMOTE,SOMO,NRAS,SYMPROD,G_SMOTE,RWO_sampling,ANS)
from aros.aros import AROS


def do_oversampling(model,X_train,y_train,**cos_para): 
    
    if model == 'original':
        return X_train,y_train
    
    elif model == 'random':
        random = RandomOverSampler()
        return random.fit_resample(X_train,y_train)
    
    elif model == 'smote':
        # smote = SMOTE(k_neighbors=3)
        smote = SMOTE()
        return smote.fit_resample(X_train,y_train)
    
    elif model == 'svm_smote':
        # svmsmote = SVMSMOTE(k_neighbors=3)
        svmsmote = SVMSMOTE()
        return svmsmote.fit_resample(X_train,y_train)
        
    elif model == 'smote_enn':
        # smote = SMOTE(k_neighbors=3)
        # smoteenn = SMOTEENN(smote=smote)
        smoteenn = SMOTEENN()
        return smoteenn.fit_resample(X_train,y_train)
    
    elif model == 'smote_tl':
        # smote = SMOTE(k_neighbors=3)
        # smotetl = SMOTETomek(smote=smote)
        smotetl = SMOTETomek()
        return smotetl.fit_resample(X_train,y_train)
    
    elif model == 'adasyn':
        # adasyn = ADASYN(n_neighbors=1,)
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
    
    elif model == 'wgan_filter':
        X_test = cos_para['X_test']
        y_test = cos_para['y_test']
        classifier = cos_para['classifier']
        return WGAN_filter(X_train,y_train,X_test,y_test,classifier)
    
    elif model == 'cos':
        # N,c,alpha,linkage,L,shrink_half,expand_half,all_safe_weight,all_safe_gen,half_safe_gen,Gaussian_scale,IR,minlabel,majlabel,visualize = get_cos_para(args[0])
        return COS(X_train,y_train,N=cos_para['N'],linkage=cos_para['linkage'])[:2]

    else:
        return 0