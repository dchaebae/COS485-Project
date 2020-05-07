import numpy as np
import matplotlib.pyplot as plt

manual = [0.0074, 0.0066, 0.0065, 0.0063, 0.0060, 0.0060, 0.0059, 0.0057,
0.0063, 0.0062, 0.0061, 0.0063, 0.0063, 0.0064, 0.0065, 0.0064,
0.0064, 0.0053, 0.0055, 0.0058, 0.0057, 0.0056, 0.0054, 0.0056,
0.0060, 0.0058, 0.0057, 0.0056, 0.0057, 0.0056, 0.0056, 0.0057, 
0.0060, 0.0057, 0.0056, 0.0057, 0.0058, 0.0058, 0.0057, 0.0057]



# Create plot
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(range(len(manual)), manual)
ax1.set_title('Autoencoder Loss by Epoch for n = 4000')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
plt.savefig('autoencoder_loss.png')