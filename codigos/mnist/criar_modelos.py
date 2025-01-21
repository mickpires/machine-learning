from tensorflow import keras
from pandas import DataFrame

def criar_modelos(normalization_data:DataFrame):
    # Cria a camada de normalização
    normalization_layer = keras.layers.Normalization(axis=-1)

    # Adapta a camada aos dados (necessário para calcular a média e o desvio padrão)
    normalization_layer.adapt(normalization_data.values.reshape(-1, 1))

    Perceptron0 = keras.models.Sequential([
        normalization_layer,
        keras.layers.Dense(2,activation='softmax')
    ])

    Perceptron1 = keras.models.Sequential([
        normalization_layer,
        keras.layers.Dense(2,activation='softmax')
    ])

    Perceptron2 = keras.models.Sequential([
        normalization_layer,
        keras.layers.Dense(2,activation='softmax')
    ])

    Perceptron3 = keras.models.Sequential([
        normalization_layer,
        keras.layers.Dense(2,activation='softmax')
    ])

    Perceptron0.compile(optimizer=keras.optimizers.SGD(learning_rate=1),loss='categorical_crossentropy',metrics=['accuracy'])
    Perceptron1.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-1),loss='categorical_crossentropy',metrics=['accuracy'])
    Perceptron2.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-2),loss='categorical_crossentropy',metrics=['accuracy'])
    Perceptron3.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3),loss='categorical_crossentropy',metrics=['accuracy'])