import collections
import pandas as pd
import tensorflow as tf 
import numpy as np
from .WGAN import *
from .TABGAN import *
from .utils import *
from COS_Funcs.utils import get_labels
from COS_Funcs.baseline.classifier import do_classification

def WGAN(X_train,y_train,target='label',classes=[0,1]):
    
    X_train,y_train = prepare_data(X_train,y_train)

    X_sample=gen_data(X_train,y_train,target,classes)
    X_train[target]=y_train
    
    X_sample=X_sample.append(X_train)
    y_sample=X_sample[target]
    X_sample=X_sample.drop(target,1)
    return X_sample.values,y_sample.values

def WGAN_filter(X_train,y_train,X_test,y_test,model,target='label',classes=[0,1]):
    
    X_train,y_train,X_test,y_test = prepare_data(X_train,y_train,X_test,y_test,target)
    preds = do_classification(X_train,y_train,X_test,model)
    init_error=error_per_class(y_test,preds,classes)
    X_out=X_train.copy()
    y_out=y_train.copy()
    minclass = [get_labels(y_train)[0]]
    X_train,y_train,X_test,y_test = prepare_data(X_train,y_train,X_test,y_test,target)
    for c in classes:
        error=init_error
        ins=y_train.index[y_train==c].tolist()
        print('GEN DATA FOR CLASS ', c)
        X_gan= gen_data(X_train, pd.DataFrame(y_train),target,set([c])) #1
        y_gan = pd.Series(X_gan[target]) #2
        X_gan=X_gan.drop(target,1) #3
        
        X_filtered,y_filtered = filter_data(X_train,y_train,X_gan,pd.Series(y_gan),
                                          X_test,init_error,c,y_test,classes)
#         X_out=X_out.append(pd.DataFrame(X_filtered))
#         y_out=pd.concat([pd.DataFrame(y_out,columns=[target]),
#                        pd.DataFrame(y_filtered,columns=[target])])
#         y_out.reset_index(inplace=True,drop=True)
#         X_out.reset_index(inplace=True,drop=True)
#         X_gan=X_filtered.append(pd.DataFrame(X_train))
#         y_gan=pd.concat([pd.DataFrame(y_filtered,columns=[target]),
#                        pd.DataFrame(y_train)])
#         clf_model=get_model(X_gan,y_gan.astype('int'))
#         preds = clf_model.predict(X_test)
#         error=error_per_class(y_test,preds,classes)
#         y_gan.reset_index(inplace=True,drop=True)
#         X_gan.reset_index(inplace=True,drop=True)


def generate_synthetic_samples(generator,class_id,headers_name,nb_instance,NOISE_DIM):
    # generete instances
    fake_data=generator(tf.random.normal([nb_instance,NOISE_DIM]))
    # prepare syhtentic dataset for export
    synthetic_data=pd.DataFrame(data=np.array(fake_data),columns=headers_name)
    synthetic_data["0"]=np.repeat(class_id,len(fake_data))
    # synthetic_data.to_csv("GAN_Synthetic_Data"+str(class_id)+".csv",index=False,header=True)
    return synthetic_data

def fake_data_generation(training_data,nb_instances_to_generate,target):
    # setting training parameters for GAN
    BATCH_SIZE=8
    NOISE_DIM=10
    learning_rate=0.001
    epochs=150
    # save column names for later
    headers_name=list(training_data.columns.values)
    headers_name=headers_name[0:-1]
    # prepre training data
    # class_id=training_data["TypeGlass"].values[0]
    class_id=training_data[target].values[0]
    print('CLASS ID',class_id)
    X=training_data.iloc[:,:-1].values.astype("float32")
    # number of features for training data 
    n_inp=X.shape[1]
    # slice training data into small batches
    train_dataset=(tf.data.Dataset.from_tensor_slices(X.reshape(X.shape[0],n_inp)).batch(BATCH_SIZE))
    # init the generator with number of features desired for the output and noise dimension
    generator=Generator(n_inp,NOISE_DIM)
    critic=Critic(n_inp)
    # Init RMSprop optimizer for the generator and the critic 
    generator_optimizer=tf.keras.optimizers.RMSprop(learning_rate)
    critic_optimizer=tf.keras.optimizers.RMSprop(learning_rate)
    # WD distance across epochs
    # Gen loss across epochs
    # Desc loss across epochs
    epoch_wasserstein=[] 
    epoch_gen_loss=[] 
    epoch_critic_loss_real=[] 
    epoch_critic_loss_fake=[]
    for epoch in range(epochs):
        batch_idx=0
        batch_wasserstein=0
        batch_gen=0
        batch_critic_real=0
        batch_critic_fake=0
        # training
        for batch in train_dataset:
            wasserstein,gen_loss,critic_loss_real,critic_loss_fake=train_step(batch,generator,critic,NOISE_DIM,generator_optimizer,critic_optimizer)
            epoch_wasserstein.append(wasserstein)
            epoch_gen_loss.append(gen_loss)
            epoch_critic_loss_real.append(critic_loss_real)
            epoch_critic_loss_fake.append(critic_loss_fake)
            batch_gen+=gen_loss
            batch_critic_real+=critic_loss_real
            batch_critic_fake+=critic_loss_fake
            batch_wasserstein+=wasserstein
            batch_idx+=1
        batch_wasserstein=batch_wasserstein/batch_idx
        batch_gen=batch_gen/batch_idx
        batch_critic_real=batch_critic_real/batch_idx
        batch_critic_fake=batch_critic_fake/batch_idx
        if epoch%50==0:
            print("Epoch %d / %d completed. Gen loss: %.8f. Desc loss_real: %.8f . Desc loss_fake: %.8f"%(epoch+1,epochs,batch_gen,batch_critic_real,batch_critic_fake))
            """nb_instances_to_generate = len(class_0["target"]) - len(class_1["target"])    """
    data=generate_synthetic_samples(generator,class_id,headers_name,nb_instances_to_generate,NOISE_DIM)
    return data

# the function to generate fake data with WGAN for the given classes 
def gen_data(X_train,y_train,target,classes):
    # count_classes=dict(y_train.value_counts())
    count_classes=collections.Counter(y_train)
    max_class=max(count_classes.values())
    print('MAX CLASS',max_class)
    new_data=pd.DataFrame()
    tmp=X_train.copy()
    tmp[target]=y_train
    
    for c in set(classes):
        training_data=tmp[tmp[target]==c]
        # get number of instances to oversample
        nb_instances_to_generate=max_class-count_classes[c]
        if nb_instances_to_generate !=0:
            syhtnetic_data=fake_data_generation(training_data,nb_instances_to_generate,target)
            syhtnetic_data.rename(columns={'0':target},inplace=True)
            syhtnetic_data[target]=c
            new_data=new_data.append(syhtnetic_data)
    return new_data
    



def TABGAN(X_train, y_train, X_test,y_test,target='label',minclass=[1]):
    minclass = [get_labels(y_train)[0]]
    X_train,y_train,X_test,y_test = prepare_data(X_train,y_train,X_test,y_test,target)
    X_sample,y_sample = run_tabgan(X_train, pd.DataFrame(y_train),X_test,
                               pd.DataFrame(y_test),target,minclass)
    
    X_sample = np.vstack((X_train.values,X_sample.values))
    y_sample = np.hstack((y_train[target],y_sample.values))
    return X_train,y_train,X_sample,y_sample


def prepare_data(X_train,y_train,X_test=None,y_test=None,target=None):
    '''
    @ Transform my baseline's input format to GAN needed input format
    @ Return: X_train,y_train or X_train,y_train,X_test,y_test
    '''
    if X_test is None and y_test is None:
        return pd.DataFrame(data=np.array(X_train)),pd.Series(y_train)
    else:
        return pd.DataFrame(data=np.array(X_train)),pd.DataFrame(y_train,columns=[target]),pd.DataFrame(data=np.array(X_test)),pd.Series(y_test)