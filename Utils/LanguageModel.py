import keras
from keras import layers, optimizers, models
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import nltk

class LanguageModel:
    def __init__(self, percentages):
        self.percentages = percentages 
        d = {}
        d['$'] = 46
        d["''"] = 1
        d['('] = 2
        d[')'] = 3
        d[','] = 4
        d['--'] = 5
        d['.'] = 6
        d[':'] = 7
        d['CC'] = 8
        d['CD'] = 9
        d['DT'] = 10
        d['EX'] = 11
        d['FW'] = 12
        d['IN'] = 13
        d['JJ'] = 14
        d['JJR'] = 15
        d['JJS'] = 16
        d['LS'] = 17
        d['MD'] = 18
        d['NN'] = 19
        d['NNP'] = 20
        d['NNPS'] = 21
        d['NNS'] = 22
        d['PDT'] = 23
        d['POS'] = 24
        d['PRP'] = 25
        d['PRP$'] = 26
        d['RB'] = 27
        d['RBR'] = 28
        d['RBS'] = 29
        d['RP'] = 30
        d['SYM'] = 31
        d['TO'] = 32
        d['UH'] = 33
        d['VB'] = 34
        d['VBD'] = 35
        d['VBG'] = 36
        d['VBN'] = 37
        d['VBP'] = 38
        d['VBZ'] = 39
        d['WDT'] = 40
        d['WP'] = 41
        d['WP$'] = 42
        d['WRB'] = 43
        d["``"] = 44
        d["#"] = 45
        self.dict = d
        network = models.Sequential()   

        network.add(layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding='Same',
        strides=1,
        input_shape=(70,47),
        activation='relu',
        kernel_initializer='he_normal',
        bias_initializer='zeros'
        ))

        network.add(layers.Conv1D(
        filters=64,
        kernel_size=7,
        padding='Same',
        strides=1,
        activation='relu',
        kernel_initializer='he_normal',
        bias_initializer='zeros'
        ))


        network.add(layers.Flatten())

        network.add(layers.BatchNormalization())

        network.add(layers.Dropout(0.3))

        network.add(layers.Dense(
        3,
        activation='softmax',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
        ))

        # learning rate and decay
        learning_rate = 0.01
        decay = 1e-3

        # optimizer
        optimizer = optimizers.Adam()

        # loss
        loss = "categorical_crossentropy"

        # metrics
        metrics = ["categorical_accuracy"]

        # dropout
        dropout = 0.3
        Name = './Utils/language_model/LanguageClassification-v1.0'

        # compile the model
        network.compile(optimizer, loss, metrics)
        #network.summary()
        network.load_weights(Name)    
        
        self.model = network
        
        return
        
        
        
        
    def predict_language(self, data):
        
        #convert data into one hot
        ConvertedData = []
        
        #get the minimum and maximum fitnet that can be get
        minimum = 0
        maximum = 0
        for p in self.percentages:
            if p<0:
                minimum += abs(p)
            else:
                maximum +=p
                
        
        for tweet in data:
            
            I = np.array([x for x,_ in tweet])
            t = np.where(I != '')
            
            #put in a zeroed list
            zeroed = np.zeros(70)
            
            #if the indivdual is not just zeros
            
            
            if t[0] != []:
                #get the tags
                tagged = np.array(nltk.pos_tag(I[:len(t[0])]))[:,1]
                #convet the tags to numbers
                tagsN = [self.dict[tag] for tag in tagged]
                zeroed[:len(tagsN)]= tagsN

            #encode into one hot
            onehotEncoded = (np.arange(47) == zeroed[...,None]-1).astype(int)
            #append indivual to new data list
            ConvertedData.append(onehotEncoded)
        #make the list an np array
        ConvertedData = np.array(ConvertedData)

        #make prediction
        predictions = self.model.predict(ConvertedData)
        #calculate the fitness
        fitness = np.sum(predictions * self.percentages, axis= 1)
        
        #add the minimum so it is at least 0
        fitness[:] += minimum
        #devide with the minimum+maximum which is the actual maximum because we add the minimum to the fitness before in order normalize
        fitness[:] = fitness[:]/(minimum+maximum)
        
        return fitness

