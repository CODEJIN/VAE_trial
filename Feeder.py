import tensorflow as tf
from threading import Thread
from collections import deque
import time, json
import numpy as np

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Feeder:
    def __init__(self, is_Training= False):
        self.is_Training = is_Training

        if self.is_Training:
            self.pattern_Queue = deque()
            pattern_Generate_Thread = Thread(target= self.Pattern_Generate)
            pattern_Generate_Thread.daemon = True
            pattern_Generate_Thread.start()

    def Pattern_Generate(self):
        patterns = tf.keras.datasets.mnist.load_data()[0][0]
        patterns = patterns.astype(np.float32) / 256.0

        while True:
            batch_Index = 0
            
            while batch_Index < patterns.shape[0]:
                if len(self.pattern_Queue) >= hp_Dict['Train']['Max_Pattern_Queue']:
                    time.sleep(0.1)
                    continue

                mnists = patterns[batch_Index:batch_Index + hp_Dict['Train']['Batch_Size']]
                mnists = np.reshape(mnists, (mnists.shape[0], -1))

                self.pattern_Queue.append({
                    'mnists': mnists
                    })

                batch_Index += 1

    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01)
        return self.pattern_Queue.popleft()

if __name__ == "__main__":
    new_Feeder = Feeder(is_Training= True)
    x = new_Feeder.Get_Pattern()
    x = np.reshape(x['mnists'], [-1, 28, 28])
    import matplotlib.pyplot as plt
    plt.imshow(x[0])
    plt.colorbar()
    plt.show()

    while True:
        time.sleep(1)
        print(new_Feeder.Get_Pattern())