import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

f_loss_x,f_loss_y = np.loadtxt('f_loss_grid.csv',delimiter=',',skiprows=1,unpack=True,dtype=np.float)
lstm_loss_x, lstm_loss_y = np.loadtxt('lstm_loss2_grid.csv',delimiter=',',skiprows=1,unpack=True,dtype=np.float)
att_loss_x, att_loss_y = np.loadtxt('att_loss_grid.csv',delimiter=',',skiprows=1,unpack=True,dtype=np.float)
plt.xlabel('plot_interval (50 units are a epoch)')
plt.ylabel('loss')
plt.plot(f_loss_x,f_loss_y, 'r', label = 'lstm_twin')
plt.plot(lstm_loss_x, lstm_loss_y,'b',label = 'lstm')
plt.plot(att_loss_x, att_loss_y,'g',label='lstm_attention')

plt.legend(loc='upper right')
plt.show()
