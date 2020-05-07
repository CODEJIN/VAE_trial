import tensorflow as tf
import json

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Encoder(tf.keras.layers.Layer):    
    def build(self, input_shapes):
        self.layer_Dict = {}

        self.layer_Dict['Label_Embedding'] = tf.keras.layers.Embedding(
            input_dim= hp_Dict['Encoder']['Label']['Nums'],
            output_dim= hp_Dict['Encoder']['Label']['Embedding']
            )
        # self.layer_Dict['Label_Embedding'] = tf.keras.layers.Lambda(
        #     lambda x: tf.one_hot(
        #         indices= x,
        #         depth= hp_Dict['Encoder']['Label']['Nums']
        #         )            
        #     )

        self.layer_Dict['Concat'] = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis= -1)
            )

        self.layer_Dict['Encoding'] = tf.keras.Sequential()
        self.layer_Dict['Encoding'].add(tf.keras.layers.Dense(
            units= hp_Dict['Encoder']['Hidden1']['Size'],
            activation= 'relu'
            ))
        self.layer_Dict['Encoding'].add(tf.keras.layers.Dropout(
            rate= hp_Dict['Encoder']['Hidden1']['Dropout_Rate']
            ))
        self.layer_Dict['Encoding'].add(tf.keras.layers.Dense(
            units= hp_Dict['Encoder']['Hidden2']['Size'],
            activation= 'tanh'
            ))
        self.layer_Dict['Encoding'].add(tf.keras.layers.Dropout(
            rate= hp_Dict['Encoder']['Hidden2']['Dropout_Rate']
            ))
        self.layer_Dict['Encoding'].add(tf.keras.layers.Dense(
            units= hp_Dict['Latent_Size'] * 2
            ))

    def call(self, inputs):
        mnists, labels = inputs
        
        labels = self.layer_Dict['Label_Embedding'](labels)
        new_Tensor = self.layer_Dict['Concat']([mnists, labels])
        new_Tensor = self.layer_Dict['Encoding'](new_Tensor)
        mean, std = tf.split(new_Tensor, num_or_size_splits= 2, axis= -1)

        return mean, std

class Latent(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, std = inputs
        std = tf.nn.softplus(std)
        return mean + std * tf.random.normal(tf.shape(std), 0, 1, dtype=mean.dtype)

class Decoder(tf.keras.layers.Layer):
    def build(self, input_shapes):
        self.layer_Dict = {}

        self.layer_Dict['Label_Embedding'] = tf.keras.layers.Embedding(
            input_dim= hp_Dict['Encoder']['Label']['Nums'],
            output_dim= hp_Dict['Encoder']['Label']['Embedding']
            )
        # self.layer_Dict['Label_Embedding'] = tf.keras.layers.Lambda(
        #     lambda x: tf.one_hot(
        #         indices= x,
        #         depth= hp_Dict['Encoder']['Label']['Nums']
        #         )            
        #     )

        self.layer_Dict['Concat'] = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis= -1)            
            )

        self.layer_Dict['Decoding'] = tf.keras.Sequential()
        self.layer_Dict['Decoding'].add(tf.keras.layers.Dense(
            units= hp_Dict['Decoder']['Hidden1']['Size'],
            activation= 'tanh'
            ))
        self.layer_Dict['Decoding'].add(tf.keras.layers.Dropout(
            rate= hp_Dict['Decoder']['Hidden1']['Dropout_Rate']
            ))
        self.layer_Dict['Decoding'].add(tf.keras.layers.Dense(
            units= hp_Dict['Decoder']['Hidden2']['Size'],
            activation= 'relu'
            ))
        self.layer_Dict['Decoding'].add(tf.keras.layers.Dropout(
            rate= hp_Dict['Decoder']['Hidden2']['Dropout_Rate']
            ))
        self.layer_Dict['Decoding'].add(tf.keras.layers.Dense(
            units= 28 ** 2,
            ))
        self.layer_Dict['Decoding'].add(tf.keras.layers.Lambda(
            lambda x: tf.clip_by_value(x, 1e-6, 1 - 1e-6),
            ))

    def call(self, inputs):
        latents, labels = inputs
        
        labels = self.layer_Dict['Label_Embedding'](labels)
        new_Tensor = self.layer_Dict['Concat']([latents, labels])
        new_Tensor = self.layer_Dict['Decoding'](new_Tensor)

        return new_Tensor

class Loss(tf.keras.layers.Layer):
    def call(self, inputs):
        labels, logits, encoder_Mean, encoder_Std = inputs

        marginal_Likelihood = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= labels, logits= logits), axis= -1)        
        kl_Divergence = 0.5 * tf.reduce_sum(encoder_Mean ** 2 + encoder_Std ** 2 - tf.math.log(encoder_Std ** 2 + 1e-8) - 1.0, axis= -1)

        marginal_Likelihood = tf.reduce_mean(marginal_Likelihood)
        kl_Divergence = tf.reduce_mean(kl_Divergence)
        
        elbo = marginal_Likelihood - kl_Divergence
        loss = -elbo

        return loss