import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
import json, os, time
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Modules import Encoder, Latent, Decoder, Loss
from Feeder import Feeder

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)


if not hp_Dict['Device'] is None:
    os.environ["CUDA_VISIBLE_DEVICES"]= hp_Dict['Device']

if hp_Dict['Use_Mixed_Precision']:    
    policy = mixed_precision.Policy('mixed_float16')
else:
    policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)


class VAE:
    def __init__(self, is_Training= False):
        self.feeder = Feeder(is_Training= is_Training)
        self.Model_Generate()

    def Model_Generate(self):
        input_Dict = {}
        layer_Dict = {}
        tensor_Dict = {}

        input_Dict['MNIST'] = tf.keras.layers.Input(
            shape= [28 ** 2,],    #MNIST
            dtype= tf.float32
            )
        input_Dict['Latent'] = tf.keras.layers.Input(
            shape= [hp_Dict['Latent_Size'],],    #MNIST
            dtype= tf.float32
            )
        input_Dict['Label'] = tf.keras.layers.Input(
            shape= [],    #MNIST
            dtype= tf.int32
            )

        layer_Dict['Encoder'] = Encoder()
        layer_Dict['Latent'] = Latent()
        layer_Dict['Decoder'] = Decoder()
        layer_Dict['Loss'] = Loss()

        tensor_Dict['Encoder_Mean'], tensor_Dict['Encoder_Std'] = layer_Dict['Encoder'](inputs= [input_Dict['MNIST'], input_Dict['Label']])
        tensor_Dict['Latent'] = layer_Dict['Latent']([tensor_Dict['Encoder_Mean'], tensor_Dict['Encoder_Std']])
        tensor_Dict['Reconstruct'] = layer_Dict['Decoder'](inputs= [tensor_Dict['Latent'], input_Dict['Label']])
        tensor_Dict['Inference'] = layer_Dict['Decoder'](inputs= [input_Dict['Latent'], input_Dict['Label']])

        tensor_Dict['Loss'] = layer_Dict['Loss']([
            input_Dict['MNIST'],
            tensor_Dict['Reconstruct'],
            tensor_Dict['Encoder_Mean'],
            tensor_Dict['Encoder_Std']
            ])

        self.model_Dict = {}
        self.model_Dict['Train'] = tf.keras.Model(
            inputs= [input_Dict['MNIST'], input_Dict['Label']],
            outputs= tensor_Dict['Loss']
            )
        self.model_Dict['Inference'] = tf.keras.Model(
            inputs= [input_Dict['Latent'], input_Dict['Label']],
            outputs= tensor_Dict['Inference']
            )

        self.model_Dict['Train'].summary()
        self.model_Dict['Inference'].summary()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate= hp_Dict['Train']['Learning_Rate'],
            beta_1= hp_Dict['Train']['ADAM']['Beta1'],
            beta_2= hp_Dict['Train']['ADAM']['Beta2'],
            epsilon= hp_Dict['Train']['ADAM']['Epsilon'],
            )
        self.checkpoint = tf.train.Checkpoint(optimizer= self.optimizer, model= self.model_Dict['Train'])

    def Train_Step(self, mnists, labels):        
        with tf.GradientTape() as tape:
            loss = self.model_Dict['Train'](inputs=[mnists, labels], training= True)
        
        gradients = tape.gradient(loss, self.model_Dict['Train'].trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_Dict['Train'].trainable_variables))
        
        return loss

    def Inference_Step(self, latents, labels):
        mnists = self.model_Dict['Inference'](inputs=[latents, labels], training= False)

        return mnists

    def Restore(self, checkpoint_File_Path= None):
        if checkpoint_File_Path is None:
            checkpoint_File_Path = tf.train.latest_checkpoint(hp_Dict['Checkpoint_Path'])

        if not os.path.exists('{}.index'.format(checkpoint_File_Path)):
            print('There is no checkpoint.')
            return

        self.checkpoint.restore(checkpoint_File_Path)
        print('Checkpoint \'{}\' is loaded.'.format(checkpoint_File_Path))

    def Train(self):
        if not os.path.exists(os.path.join(hp_Dict['Inference_Path'], 'Hyper_Parameters.json')):
            os.makedirs(hp_Dict['Inference_Path'], exist_ok= True)
            with open(os.path.join(hp_Dict['Inference_Path'], 'Hyper_Parameters.json').replace("\\", "/"), "w") as f:
                json.dump(hp_Dict, f, indent= 4)

        def Save_Checkpoint():
            os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)
            self.checkpoint.save(
                os.path.join(
                    hp_Dict['Checkpoint_Path'],
                    'S_{}.CHECKPOINT.H5'.format(self.optimizer.iterations.numpy())
                    ).replace('\\', '/')
                )
           
        def Run_Inference():
            latents = None
            labels = None

            labels = np.tile(np.arange(0, 10), 10).astype(np.int32)

            # if hp_Dict['Latent_Size'] == 2:
            #     latents = []                
            #     for x in np.arange(-1.0, 1.0 + 0.1, 0.1):
            #         for y in np.arange(-1.0, 1.0 + 0.1, 0.1):
            #             latents.append(np.array([x,y], dtype= np.float32))
            #     latents = np.stack(latents, axis= 0)
            #     labels = np.zeros(shape=(latents.shape[0]), dtype= np.int32)  + 5
                        
            self.Inference(latents= latents, labels= labels)

        if hp_Dict['Train']['Initial_Inference']:
            Run_Inference()

        while True:
            start_Time = time.time()

            loss = self.Train_Step(**self.feeder.Get_Pattern())

            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Step: {}'.format(self.optimizer.iterations.numpy()),  #Total and Adversarial                
                'Loss: {:0.5f}'.format(loss)
                ]
            print('\t\t'.join(display_List))
            with open(os.path.join(hp_Dict['Inference_Path'], 'log.txt'), 'a') as f:
                f.write('\t'.join([
                '{:0.3f}'.format(time.time() - start_Time),
                '{}'.format(self.optimizer.iterations.numpy()),
                '{:0.5f}'.format(loss)
                ]) + '\n')

            if self.optimizer.iterations.numpy() % (hp_Dict['Train']['Checkpoint_Save_Timing']) == 0:
                Save_Checkpoint()
            
            if self.optimizer.iterations.numpy() % (hp_Dict['Train']['Inference_Timing']) == 0:
                Run_Inference()

            start_Time = time.time()

    def Inference(self, latents= None, labels= None, file_Label= None):        
        if latents is None and labels is None:
            batch_Size = 1
        elif not latents is None:
            batch_Size = latents.shape[0]
        elif not labels is None:
            batch_Size = labels.shape[0]

        if latents is None:
            latents = np.random.normal(size=(batch_Size, hp_Dict['Latent_Size']))
        if labels is None:
            labels = np.random.randint(0, hp_Dict['Encoder']['Label']['Nums'], size= (batch_Size,), dtype= np.int32)

        mnists = self.Inference_Step(latents= latents, labels= labels)

        self.Export_Inference(
            mnists= mnists,
            file_Label= None or datetime.now().strftime("%Y%m%d.%H%M%S")
            )

    def Export_Inference(self, mnists, file_Label):
        os.makedirs(hp_Dict['Inference_Path'], exist_ok= True)

        batch_Size = mnists.shape[0]
        axis_Count = int(np.ceil(np.sqrt(batch_Size)))

        mnists = np.reshape(mnists, [batch_Size, 28, 28])

        fig = plt.figure(figsize=(10, 10))
        new_Gridspec = gridspec.GridSpec(axis_Count, axis_Count)
        new_Gridspec.update(wspace=0.025, hspace=0.025)
        for index in range(batch_Size):
            mnist = mnists[index]
            ax = plt.subplot(new_Gridspec[index])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.imshow(mnist, cmap='gray')

        plt.tight_layout(fig)
        plt.savefig(
            os.path.join(hp_Dict['Inference_Path'], '{}.PNG'.format(file_Label)).replace("\\", "/")
            )
        plt.close(fig)


if __name__ == "__main__":
    new_Model = VAE(is_Training= True)
    new_Model.Restore()
    new_Model.Train()