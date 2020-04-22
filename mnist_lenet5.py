# imports 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(16)
np.random.seed(16)

# get train data
mnist = torch.load('/tigress/dkchae/MNIST/processed/training.pt')
trainimages = mnist.data
trainlabels = mnist.targets
IMAGE_SAMPLES = trainimages.shape[0]

# get test data
mnist_test = torch.load('/tigress/dkchae/MNIST/processed/test.pt')
testimages = mnist_test.data
testlabels = mnist_test.targets

# split up the training set into
indices = []
plot_samples = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 60000])
for i in range(plot_samples.shape[0]):
  indices.append(np.random.choice(range(IMAGE_SAMPLES), plot_samples[i], replace=False))

from copy import deepcopy

# Taken from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
              print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss

class LeNet5(nn.Module):

    # definition of each neural network layer
    def __init__(self):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.S2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        ####### Complete the defition of C3 and S4 ##########
        # C3 is a convolutional layer with 16 5x5 kernels
        # S4 is a max pooling layer with 2x2 kernel and stride 2

        self.C3 = nn.Conv2d(6, 16, kernel_size=(5,5))
        self.S4 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        
        #####################################################

        self.C5 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.F6 = nn.Linear(120, 84)

        # output layer
        self.OL = nn.Linear(84, 10)

    # definition of the forward pass
    def forward(self, x):
      
        x = torch.tanh(self.C1(x))
        x = self.S2(x)
        x = torch.tanh(self.C3(x))
        x = self.S4(x)
        x = torch.tanh(self.C5(x))

        ##################################################################################

        # convert (batch, 120, 1, 1) feature maps to 1d features of size (batch, 120)
        x = x.view(x.size(0), -1) 

        # pass the activation to the fully connected layer F6 followed by a tanh activation
        # output size changes from (batch, 120) to (batch, 86)
        x = torch.tanh(self.F6(x))

        # pass the activation to the final fully connected layer OL followed by a tanh activation
        # output size changes from (batch, 86) to (batch, 10)
        x = torch.tanh(self.OL(x))
        
        return x

    def _initialize_weights(self, model_state_dict):
        self.C1.weight = torch.nn.Parameter(deepcopy(model_state_dict['C1.weight']))
        self.C1.bias = torch.nn.Parameter(deepcopy(model_state_dict['C1.bias']))

        self.C3.weight = torch.nn.Parameter(deepcopy(model_state_dict['C3.weight']))
        self.C3.bias = torch.nn.Parameter(deepcopy(model_state_dict['C3.bias']))

        self.C5.weight = torch.nn.Parameter(deepcopy(model_state_dict['C5.weight']))
        self.C5.bias = torch.nn.Parameter(deepcopy(model_state_dict['C5.bias']))
        
        
        self.F6.weight = torch.nn.Parameter(deepcopy(model_state_dict['F6.weight']))
        self.F6.bias = torch.nn.Parameter(deepcopy(model_state_dict['F6.bias']))

def train(train_set, test_set, model_state_dict = None, batchsize = 32, nepoch=1000, patience=5):

  train_images, train_labels = train_set
  test_images, test_labels = test_set


  ntrain = train_images.shape[0];  # number of training examples

  lenet5 = LeNet5()
  if model_state_dict is not None:
    print('initialize with weights!')
    lenet5._initialize_weights(model_state_dict)

  t_start = time.time()
  # use SGD optimizer
  optimizer = optim.SGD(lenet5.parameters(), lr=0.05)

  # Put in Early Stopping
  early_stopping = EarlyStopping(patience=patience)

  train_losses = []

  for iepoch in range(nepoch):
      random_order = np.arange(ntrain)
      np.random.shuffle(random_order)
      batch_num = 0

      while batch_num < ntrain:
          batchindices = random_order[batch_num:min(ntrain, batch_num + batchsize)]
          trainlabels_iter = train_labels[batchindices]

          # normalize input images
          imgs = torch.zeros([batchindices.shape[0], 1, 32, 32])
          imgs[:, 0, 2: -2, 2: -2] = train_images[batchindices].float() / 255.

          # before the forward pass, clean the gradient buffers of all parameters
          optimizer.zero_grad()

          # forward pass
          out = lenet5(imgs)
          
          # Cross Entropy
          loss = torch.nn.CrossEntropyLoss()(out, trainlabels_iter)

          # record training loss
          train_losses.append(loss.item())

          # backward pass
          loss.backward()

          # update parameters using SGD
          optimizer.step()
          
          # update batch number
          batch_num += batchsize
        
      train_loss = np.mean(train_losses)

      early_stopping(train_loss, lenet5)

      if early_stopping.early_stop:
        print("Early stopping at Epoch %d, batch %d of batchsize %d" % (iepoch, 
                                                                        int(batch_num/batchsize),
                                                                        batchsize))
        break

      # reset the list for next iteration
      train_losses = []

  # RESULTS
  # normalize input images 
  imgs = torch.zeros([ntrain, 1, 32, 32])
  imgs[:, 0, 2: -2, 2: -2] = train_images.float() / 255.
  out = lenet5(imgs)
  # calculate error rate and loss for plot
  pred = torch.argmax(out, dim=1)
  print(pred[:10])
  print(train_labels[:10])

  err_train = torch.mean((pred != train_labels).float())

  # normalize input images
  imgs = torch.zeros([len(test_images), 1, 32, 32])
  imgs[:, 0, 2: -2, 2: -2] = test_images.float() / 255.
  # calculate error rate and loss for plot
  out = lenet5(imgs)
  pred = torch.argmax(out, dim=1)
  err_test = torch.mean((pred != test_labels).float())

  return err_train, err_test

# RUN THE FIRST TEST
print('=============Running Baseline===========')
test_tuple = (testimages, testlabels)
err_train_all = np.zeros(len(indices))
err_test_all = np.zeros(len(indices))
for i in range(len(indices)):
  print('Running %d of %d' % (i+1, len(indices)))
  train_tuple = (trainimages[indices[i]], trainlabels[indices[i]])
  x = train(train_tuple, test_tuple)
  print(x)
  err_train_all[i], err_test_all[i] = x
print(err_train_all)
print(err_test_all)


import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = x.view(x.size(0), 1, 32, 32)
    return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return x

  
class Unflatten(torch.nn.Module):
    def forward(self, x):
        x = x.view(1, 120, 1, 1) 
        return x


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.S2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices = True)

        self.C3 = nn.Conv2d(6, 16, kernel_size=(5,5))
        self.S4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices = True)

        self.C5 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.F6 = nn.Linear(120, 84)


        # self.encoder = nn.Sequential(
        #   self.C1,
        #   nn.Tanh(),
        #   self.S2,
        #   self.C3,
        #   nn.Tanh(),
        #   self.S4,
        
        #   self.C5,
        #   nn.Tanh(),
        #   Flatten(),
        #   self.F6,

        # )

        self.r_F6 = nn.Linear(84, 120)
        self.r_C5 = nn.ConvTranspose2d(120, 16, kernel_size=(5, 5))

        self.r_S4 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)
        self.r_C3 = nn.ConvTranspose2d(16, 6, kernel_size=(5,5))

        self.r_S2 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)
        self.r_C1 = nn.ConvTranspose2d(6, 1, kernel_size=(5, 5))


        # self.decoder = nn.Sequential(
        #     self.r_F6,
        #     Unflatten(),
        #     nn.Tanh(),
        #     self.r_C5,  # b, 16, 5, 5
        #     self.r_S4,
        #     self.r_C3,  # b, 8, 15, 15
        #     nn.Tanh(),
        #     self.r_S2,
        #     nn.Tanh(),
        #     self.r_C1  # b, 1, 28, 28

        # )

    def forward_encode(self, x):
      x = torch.tanh(self.C1(x))
      x, self.inds1 = self.S2(x)

      x = torch.tanh(self.C3(x))
      x, self.inds2 = self.S4(x)

      x = torch.tanh(self.C5(x))
      temp = Flatten()
      x = temp.forward(x)
      x = self.F6(x)
      return x

    def forward_decode(self, x):
      x = self.r_F6(x)
      temp = Unflatten()
      x = temp.forward(x)
      x = torch.tanh(x)
      
      x = self.r_C5(x)
      x = self.r_S4(x, self.inds2)
      x = torch.tanh(x)
      x = self.r_C3(x)
      
      x = self.r_S2(x, self.inds1)
      x = torch.tanh(x)
      x = self.r_C1(x)
      return x

    def forward(self, x):
      x = self.forward_encode(x)
      x = self.forward_decode(x)
      return x

num_epochs = 10
batch_size = 128
learning_rate = 1e-3

model = autoencoder()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

print('==========Train Autoencoder==========')
for epoch in range(num_epochs):
    epoch_loss = 0

    for i in torch.randperm(trainimages.shape[0]):
        img = trainimages[i]
        img2 = torch.zeros([1, 1, 32, 32])
        img2[:, 0, 2: -2, 2: -2] = img.float() / 255.
        img = Variable(img2)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        epoch_loss += loss.data.item()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    epoch_loss /= trainimages.shape[0]
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, epoch_loss))
    if epoch % 2 == 0 or epoch == num_epochs - 1:
        pic = to_img(output.cpu().data)

        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.imshow(pic.reshape(32, 32))
        plt.subplot(122)
        plt.imshow(img.reshape(32, 32))
        plt.grid(False)
        plt.show()

model_state_dict = deepcopy(model.state_dict())

print('=========Running Initialized=========')
test_tuple = (testimages, testlabels)
err_train_all_initialized = np.zeros(len(indices))
err_test_all_initialized = np.zeros(len(indices))
for i in range(len(indices)):
  print('Running %d of %d' % (i+1, len(indices)))
  train_tuple = (trainimages[indices[i]], trainlabels[indices[i]])
  x= train(train_tuple, test_tuple, model_state_dict)
  print(x)
  err_train_all_initialized[i], err_test_all_initialized[i] = x
print(err_train_all_initialized)
print(err_test_all_initialized)


log_plot_samples = np.log2(plot_samples)

data = (err_train_all, err_test_all)
data_initialized = (err_train_all_initialized, err_test_all_initialized)
groups = ("Training", "Test")

# Create plot
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(log_plot_samples, data[0], 'go', label=groups[0])
ax1.plot(log_plot_samples, data[1], 'bo', label=groups[1])
ax1.set_title('Training and Test Error Without Unlabeled Initialization')
ax1.legend(loc='upper right')
ax1.set_xlabel('Log2 Number of Samples')
ax1.set_ylabel('Proportion Error Rate')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(log_plot_samples, data_initialized[0], 'go', label=groups[0])
ax2.plot(log_plot_samples, data_initialized[1], 'bo', label=groups[1])
ax2.set_title('Training and Test Error With Unlabeled Initialization')
ax2.legend(loc='upper right')
ax2.set_xlabel('Log2 Number of Samples')
ax2.set_ylabel('Proportion Error Rate')

plt.savefig('errors.png')