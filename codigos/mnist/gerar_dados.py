import pandas as pd
import numpy as np

np.random.seed(50)
detection = np.round(np.linspace(460,590,200),2)
color = []
for detect in detection:
    if detect < 540:
        color.append('azul')
    else:
        color.append('verde')

data = pd.DataFrame({'comprimento de onda':detection,'cor':color})
num = 4

match num:
    case 0:
        data.to_csv("dados/dados_perfeito.csv")
        print(num)
    case 1: 
        data['comprimento de onda'] += np.random.normal(0,5,len(data['comprimento de onda']))
        data.to_csv("dados/dados_imperfeito_normal.csv")
        print(num)
    case 2: 
        data['comprimento de onda'] += np.random.uniform(-5,5,len(data['comprimento de onda']))
        data.to_csv("dados/dados_imperfeito_uniforme.csv")
        print(num)
    case 3: 
        data['comprimento de onda'] += np.random.exponential(10,len(data['comprimento de onda']))
        data.to_csv("dados/dados_imperfeito_exponencial.csv")
        print(num)
    case 4: 
        data['comprimento de onda'] += np.random.poisson(0,len(data['comprimento de onda']))
        data.to_csv("dados/dados_imperfeito_poisson.csv")
        print(num)