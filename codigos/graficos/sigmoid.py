import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x): return 1/(1+np.exp(-x))

def ReLu(x):
    y = np.array([])
    for value in x:
        if value > 0:
            y = np.hstack((y,value))
        else:
            y = np.hstack((y,0))
    return y

def leakyReLu(x,alpha):
    y = np.array([])
    for value in x:
        if value > 0:
            y = np.hstack((y,alpha*value))
        else:
            y = np.hstack((y,0))
    return y

def ELU(x,alpha):
    y = np.array([])
    for value in x:
        if value > 0:
            y = np.hstack((y,value))
        else:
            y = np.hstack((y,alpha*np.exp(value)-1))
    return y

def swish(x,alpha):
    return x*sigmoid(alpha*x)


x = np.linspace(-10,10,1000)

y = swish(x,.1)
nome_função = "$swish(x)$"
plt.plot(x,y)
plt.ylabel(nome_função)
plt.xlabel("x")
plt.grid()
plt.show()
print()
plt.plot(x,y)
plt.ylabel(nome_função)
plt.xlabel("x")
plt.grid()
nome = input('salvar? ')

if nome != "":
    plt.savefig(f"/home/mickael/Documentos/faculdade/tcc/machine-learning/imagens/{nome}")