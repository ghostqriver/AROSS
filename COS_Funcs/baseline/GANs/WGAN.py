import tensorflow as tf 
from tensorflow.keras.layers import Dense

# generator
class Generator(tf.keras.Model):
    def __init__(self,n_inp,n_noise,n_hid=128):
        super().__init__()
        init=tf.keras.initializers.GlorotUniform
        self.input_layer=Dense(units=n_noise,kernel_initializer=init)
        self.hidden_layer=Dense(units=n_hid,activation="relu",kernel_initializer=init)
        self.output_layer=Dense(units=n_inp,activation="sigmoid",kernel_initializer=init)
    def call(self,inputs):
        x=self.input_layer(inputs)
        x=self.hidden_layer(x)
        return self.output_layer(x)
    
# critic   
class Critic(tf.keras.Model):
    def __init__(self,n_inp,n_hid=128):
        super().__init__()
        init=tf.keras.initializers.GlorotUniform
        self.input_layer=Dense(units=n_inp,kernel_initializer=init)
        self.hidden_layer=Dense(units=n_hid,activation="relu",kernel_initializer=init)
        self.logits=Dense(units=1,activation=None,kernel_initializer=init)
    
    def call(self,inputs):
        x=self.input_layer(inputs)
        x=self.hidden_layer(x)
        return self.logits(x)

# @tf.function
def train_step(real_data,gen,critic,noise_dim,generator_optimizer,critic_optimizer):
    batch_size=real_data.shape[0]# gaussian noise :z
    noise=tf.random.normal([batch_size,noise_dim])
    with tf.GradientTape() as gen_tape,tf.GradientTape() as critic_tape:# x' = G(z)
        fake_data=gen(noise,training=True)# s^ = c(x)
        real_output=critic(real_data,training=True)# s_ = c(x')
        fake_output=critic(fake_data,training=True)
        critic_loss=tf.reduce_mean(fake_output)-tf.reduce_mean(real_output)
        critic_loss_real=tf.reduce_mean(real_output)
        critic_loss_fake=tf.reduce_mean(fake_output)# G loss fucntion is the critic's output for fake data -(s_)
        gen_loss=-tf.reduce_mean(fake_output)
    wasserstein=tf.reduce_mean(real_output)-tf.reduce_mean(fake_output)# calculate gradients for gen and critic to update them weights
    gradients_of_generator=gen_tape.gradient(gen_loss,gen.trainable_variables)
    gradients_of_critic=critic_tape.gradient(critic_loss,critic.trainable_variables)# update gen and critic weights 
    generator_optimizer.apply_gradients(zip(gradients_of_generator,gen.trainable_variables))
    critic_optimizer.apply_gradients(zip(gradients_of_critic,critic.trainable_variables))
    tf.group(*(var.assign(tf.clip_by_value(var,-0.01,0.01)) for var in critic.trainable_variables)) 
    return wasserstein,gen_loss,critic_loss_real,critic_loss_fake