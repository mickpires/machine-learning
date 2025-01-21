import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import StratifiedShuffleSplit

def mapa_calor(y_pred,y_test,titulo,encoder:OneHotEncoder):
    y_pred = np.round(y_pred,0)
    confusion_matrix(encoder.inverse_transform(y_test),encoder.inverse_transform(y_pred))
    labels = ('azul','verde')

    sns.heatmap(confusion_matrix(encoder.inverse_transform(y_test),
                                encoder.inverse_transform(y_pred)),
                                xticklabels=labels,
                                yticklabels=labels,annot=True,
                                cmap='cividis',
                                )
    plt.title(titulo)
    plt.savefig("imagens/"+titulo+'.png')


def acuracia(y_pred,y_test):
    y_pred = np.argmax(y_pred,axis=1)
    y_test = np.argmax(y_test,axis=1)
    accuracy = np.mean(y_test ==y_pred)
    return accuracy

def dividir_dados(dados):
    # Cria a camada de normalização
    normalization_layer = keras.layers.Normalization(axis=-1)

    # Adapta a camada aos dados (necessário para calcular a média e o desvio padrão)
    normalization_layer.adapt(dados['comprimento de onda'].values.reshape(-1, 1))
    encoder = OneHotEncoder()
    split = StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=42)

    for train_index, test_index in split.split(dados,dados['cor']):
        strat_train_set = dados.loc[train_index]
        strat_test_set = dados.loc[test_index]
    
    x_train = strat_train_set['comprimento de onda'].values.reshape(-1,1)
    y_train = encoder.fit_transform(strat_train_set['cor'].values.reshape((-1,1))).toarray()
    x_test= strat_test_set['comprimento de onda'].values.reshape(-1,1)
    y_test = encoder.fit_transform(strat_test_set['cor'].values.reshape((-1,1))).toarray()

    return x_train,y_train,x_test,y_test

def criar_MLP(x_train,dados,learning_rate):
    # Cria a camada de normalização
    normalization_layer = keras.layers.Normalization(axis=-1)

    # Adapta a camada aos dados (necessário para calcular a média e o desvio padrão)
    normalization_layer.adapt(dados['comprimento de onda'].values.reshape(-1, 1))
    input = keras.layers.Input(shape=x_train.shape[1:])
    normalized_input = normalization_layer(input)
    hidden1 = keras.layers.Dense(100,activation='tanh')(normalized_input)
    hidden2 = keras.layers.Dense(50,activation='tanh')(hidden1)
    outputs = keras.layers.Dense(2,activation='softmax')(hidden2)

    MLP = keras.models.Model(inputs=[input],outputs=[outputs])
    MLP.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=keras.optimizers.SGD(learning_rate=learning_rate))
    return MLP

