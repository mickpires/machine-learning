import numpy as np

H = np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]])
comp = np.array([[1,0],[0,1]])

print(H@comp - comp@H)