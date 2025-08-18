import torch 
import sklearn 
from sklearn.metrics import accuracy_score 
import random
from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn 
from torch import tensor 
from torch.optim import Adam
import matplotlib.pyplot as plt 
from torch.nn import functional 
from mlxtend.data import loadlocal_mnist


images_path = r"C:\Users\arthu\OneDrive\Desktop\SDAE Project\train-images.idx3-ubyte"
labels_path = r"C:\Users\arthu\OneDrive\Desktop\SDAE Project\train-labels.idx1-ubyte"
raw_x, raw_y = loadlocal_mnist(images_path, labels_path)
raw_x = raw_x.astype(np.float64)
raw_x = raw_x / 256.0
X, y = map(tensor, (raw_x, raw_y))

first_image_train = raw_x[0,:]
first_label_train = raw_y[0]
print(f"{first_image_train.dtype=}")
print(f"{first_label_train.dtype=}") 
print(f"{first_image_train=}")
print(f"{first_label_train=}")
first_image_min, first_image_max = first_image_train.min(), first_image_train.max()
print(f"{first_image_min=}, {first_image_max=}") 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

#print(f"{X_train.shape=}, {y_train.shape=}") 
print(f"{X_test.shape=}, {y_test.shape=}") 
print(X_test[0].shape)

class autoenc(nn.Module):

    def __init__(self, n_in, n_hid, n_out): 
        super().__init__()
        self.encoder = [nn.Linear(n_in, n_hid), nn.ReLU(), nn.Linear(n_hid, n_hid // 2), nn.ReLU(), nn.Linear(n_hid // 2, n_out), nn.ReLU()]
        self.decoder = [nn.Linear(n_out, n_hid // 2), nn.ReLU(), nn.Linear(n_hid // 2, n_hid), nn.ReLU(), nn.Linear(n_hid, n_in), nn.Sigmoid()]

    def forward(self, x): 
        for i in self.encoder: 
            x = i(x)
        for i in self.decoder: 
            x = i(x)
        return x



def loss_func(pred, target): 
    return functional.binary_cross_entropy(pred, target)

automodel = autoenc(784, 392, 64)

"""Remember to float() your inputs or you will get a type eror as nn.linear is picky"""

num_epochs = int(input('Num EPOCHS: '))
batch_size = 64
lr = float(input('Learning Rate: '))

loss_arr = []

def train(): 
    for epoch in range(num_epochs):
        for batch_num in range(len(X_train)//batch_size): 
            start_point = batch_num * batch_size 
            end_point = start_point + batch_size 
            batch_data = X_train[start_point : end_point]
            pred_data = automodel(batch_data.float())
            loss = loss_func(pred_data, batch_data.float())
            loss.backward()

            if loss > 0: 
                with torch.no_grad(): 
                    for i in automodel.encoder: 
                        if hasattr(i, 'weight'): 
                            i.weight -= i.weight.grad * lr 
                            i.bias -= i.bias.grad * lr
                            i.weight.grad.data.zero_()
                            i.bias.grad.data.zero_()
                            #print(i.bias.grad.data)
                    

                    for i in automodel.decoder: 
                        if hasattr(i, 'weight'): 
                            i.weight -= i.weight.grad * lr 
                            i.bias -= i.bias.grad * lr 
                            i.weight.grad.data.zero_()
                            i.bias.grad.data.zero_()
                            #print(i.bias.grad.data)
                        
        print("EPOCH {}, LOSS {}".format(epoch, loss))
        loss_arr.append(loss)  
        
train()


def noise(img, thresh) -> np.ndarray:
    rand_floats = np.random.rand(len(img))
    flip_indices = np.where(rand_floats < thresh)[0]
    clone_img = img.detach().clone()
    clone_img[flip_indices] = 1.0 - clone_img[flip_indices]
    """ # slow
    for i in np.arange(len(vec)):
        if rand_floats[i] < k:
            # noisify this pixel: flip it
            result[i] = 1.0 - vec[i]
    """
    return clone_img

def denoise(noise, img):
    
    reshaped = torch.reshape(noise, img.shape)
    pred = automodel(reshaped.float())
    return(pred)

def noise_plot(orig, thresh): 

    fig, axarr = plt.subplots(1,3)

    axarr[0].imshow(torch.reshape(orig, (28, 28)).detach().numpy(),cmap = "binary")
    axarr[0].set_title("Original")

    noised_img = noise(orig, thresh)

    axarr[1].imshow(torch.reshape(noised_img, (28, 28)).detach().numpy(),cmap = "binary")
    axarr[1].set_title("Noised")

    axarr[2].imshow(torch.reshape(denoise(noised_img, orig), (28,28)).detach().numpy(), cmap = "binary")
    axarr[2].set_title("Rebuilt")
    
    plt.show()

def reshape(i): 
    re = torch.reshape(i, (28, 28))
    return(re)

predictions = automodel(X_test.float())
i = random.randint(1,100)

pred1 = reshape(predictions[i]).detach().numpy()
val1 = reshape(X_test[i])

img1_title = f"image-{i}"
img2_title = f"image-{i+10}"

#plot original vs predictions
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(val1,cmap = "binary")
axarr[0].set_title("Original " + img1_title)
#import code; code.interact(local=dict(globals(), **locals()))
axarr[1].imshow(pred1,cmap = "binary")
axarr[1].set_title("Predicted/rebuilt " + img1_title)

plt.show()


def conv(arr): 
    conv_arr = []
    for i in arr: 
        conv_arr.append(i.detach().numpy())
    return conv_arr

plt.plot(range(num_epochs), conv(loss_arr))
plt.suptitle('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

noise_plot(X_test[0], 0.1)