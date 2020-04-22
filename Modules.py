import tensorflow as tf
import json

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Encoder(tf.keras.layers.Layer):    
    def build(self, input_shapes):
        self.layer = tf.keras.Sequential()
        self.layer.add(tf.keras.layers.Dense(
            units= hp_Dict['Encoder']['Hidden1']['Size'],
            activation= 'relu'
            ))
        self.layer.add(tf.keras.layers.Dropout(
            rate= hp_Dict['Encoder']['Hidden1']['Dropout_Rate']
            ))
        self.layer.add(tf.keras.layers.Dense(
            units= hp_Dict['Encoder']['Hidden2']['Size'],
            activation= 'tanh'
            ))
        self.layer.add(tf.keras.layers.Dropout(
            rate= hp_Dict['Encoder']['Hidden2']['Dropout_Rate']
            ))
        self.layer.add(tf.keras.layers.Dense(
            units= hp_Dict['Latent_Size'] * 2
            ))

    def call(self, inputs):
        new_Tensor = self.layer(inputs)
        mean, std = tf.split(new_Tensor, num_or_size_splits= 2, axis= -1)        

        return mean, std

class Latent(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, std = inputs
        std = tf.nn.softplus(std)
        return mean + std * tf.random.normal(tf.shape(std), 0, 1, dtype=mean.dtype)

class Decoder(tf.keras.layers.Layer):
    def build(self, input_shapes):
        self.layer = tf.keras.Sequential()
        self.layer.add(tf.keras.layers.Dense(
            units= hp_Dict['Decoder']['Hidden1']['Size'],
            activation= 'tanh'
            ))
        self.layer.add(tf.keras.layers.Dropout(
            rate= hp_Dict['Decoder']['Hidden1']['Dropout_Rate']
            ))
        self.layer.add(tf.keras.layers.Dense(
            units= hp_Dict['Decoder']['Hidden2']['Size'],
            activation= 'relu'
            ))
        self.layer.add(tf.keras.layers.Dropout(
            rate= hp_Dict['Decoder']['Hidden2']['Dropout_Rate']
            ))
        self.layer.add(tf.keras.layers.Dense(
            units= 28 ** 2,
            activation= 'sigmoid'
            ))

    def call(self, inputs):
        return self.layer(inputs)

class Loss(tf.keras.layers.Layer):
    def call(self, inputs):
        labels, logits, encoder_Mean, encoder_Std = inputs

        marginal_Likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels= labels,
            logits= logits
            ))
        kl_Divergence = -0.5 * tf.reduce_sum(
            1 + encoder_Std - encoder_Mean ** 2 - tf.exp(encoder_Std)
            )

        elbo_Loss = marginal_Likelihood + kl_Divergence

        return elbo_Loss