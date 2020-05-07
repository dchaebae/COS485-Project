import numpy as np
import matplotlib.pyplot as plt

manual_train = np.array([[0.0, 0.004000000189989805, 0.004000000189989805, 0.004999999888241291], 
[0.00800000037997961, 0.05999999865889549, 0.0, 0.003000000026077032] ,
[0.00800000037997961, 0.004000000189989805, 0.0, 0.006000000052154064],
[0.01600000075995922, 0.0, 0.0, 0.004000000189989805],
[0.01600000075995922, 0.06800000369548798, 0.004000000189989805, 0.004000000189989805],
[0.00800000037997961, 0.004000000189989805, 0.004000000189989805, 0.007000000216066837]])

manual_test = np.array([[0.20569999516010284, 0.10540000349283218, 0.0940999984741211, 0.06430000066757202], 
[0.23469999432563782, 0.16439999639987946, 0.09390000253915787, 0.07280000299215317, ],
[0.19910000264644623, 0.12210000306367874, 0.09440000355243683, 0.0763000026345253,],
[0.20340000092983246, 0.10209999978542328, 0.09220000356435776, 0.07410000264644623],
[0.20350000262260437, 0.1647000014781952, 0.09480000287294388, 0.0689999982714653],
[0.20069999992847443, 0.11659999936819077, 0.09989999979734421, 0.06870000064373016]])


flip_num = 5
plot_samples = np.array([125, 250, 500, 1000])
log_plot_samples = np.log2(plot_samples)

groups = ['Flip %d' % (d) for d in range(flip_num+5)]

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
for i in range(flip_num + 1):
  ax1.plot(log_plot_samples, manual_train[i], label=groups[i])
ax1.set_title('Training Error')
ax1.legend(loc='upper right')
ax1.set_xlabel('Log2 Number of Samples')
ax1.set_ylabel('Proportion Error Rate')
plt.savefig('train_error_bugfix.png')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
for i in range(flip_num + 1):
  ax1.plot(log_plot_samples, manual_test[i], label=groups[i])
ax1.set_title('Test Error')
ax1.legend(loc='upper right')
ax1.set_xlabel('Log2 Number of Samples')
ax1.set_ylabel('Proportion Error Rate')
plt.savefig('test_error_bugfix.png')