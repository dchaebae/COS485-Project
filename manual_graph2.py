import numpy as np
import matplotlib.pyplot as plt

manual = np.array([[0.016,0.016,0.007,0.013,0.0085,0.004,
  0.002125,0.0029375,  0.00159375, 0.00146667],
 [0.016,      0.014,      0.006,      0.014,      0.007,      0.005,
  0.004125,   0.0019375,  0.0015625,  0.00163333],
 [0.016 ,      0.016    ,  0.008   ,   0.008 ,     0.009   ,   0.003,
  0.002625 , 0.0026875  ,0.00171875 ,0.00133333],
 [0.016   ,   0.012   ,   0.004   ,   0.006   ,   0.008  ,    0.00425,
  0.003   ,   0.0030625 , 0.001625  , 0.00168333],
 [0.012  ,    0.016   ,   0.002  ,    0.015  ,   0.011    ,  0.004,
  0.002625 ,  0.0028125  ,0.0015625,  0.00155   ],
 [0.012 ,     0.016 ,     0.008   ,   0.011  ,    0.0105, 0.00525,
  0.003   ,   0.0025625,  0.00184375, 0.00153333]])


flip_num = 5
plot_samples = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 60000])
log_plot_samples = np.log2(plot_samples)

groups = ['Flip %d' % (d) for d in range(flip_num+5)]

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
for i in range(flip_num + 1):
  ax1.plot(log_plot_samples, manual[i], label=groups[i])
ax1.set_title('Training Error')
ax1.legend(loc='upper right')
ax1.set_xlabel('Log2 Number of Samples')
ax1.set_ylabel('Proportion Error Rate')
plt.savefig('train_error_mod.png')