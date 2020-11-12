import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from .utils.model import Generator,Discriminator
from .utils import loss
import os
from hyperparameters import lr,batch_size,beta1,beta2

train_on_gpu =  torch.cuda.is_available()
def get_data_loader(image_dir,image_size = 256, batch_size = batch_size,
                     num_workers = 0):
    '''
    Preprocessing of Images and getting the data ready for Model
    :param image_type: Monet or Real Photo
    :param image_size: To resize the image to one particular
    :param batch_size: batch size for batch processing of the images
    :param num_workers: For parallel processing purpose
    :return: data loader (iterative)
    '''
    transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),
                                    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

    train_path = os.path.join('./',image_dir)
    # define datasets using Image Folder
    train_dataset = datasets.ImageFolder(train_path,transform)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True,
                              num_workers = num_workers)
    return train_loader

def train(n_epochs):
    dataloader_monet = get_data_loader(image_dir = 'monet')
    dataloader_photo = get_data_loader(image_dir = 'photo')

    G_XtoY = Generator()
    G_YtoX = Generator()
    D_X = Discriminator()
    D_Y = Discriminator()
    print_every = 10
    if train_on_gpu:
        print("Moving Model to GPU")
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()

    

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())

    # creating optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params,lr,(beta1,beta2))
    d_x_optimizer = optim.Adam(D_X.parameters(), lr, (beta1, beta2))
    d_y_optimizer = optim.Adam(D_Y.parameters(), lr, (beta1, beta2))

    # batches per epoch
    iter_monet = iter(dataloader_monet)
    iter_photo = iter(dataloader_photo)

    batches_per_epoch = min(len(iter_monet), len(iter_photo))
    losses = []

    for epoch in range(n_epochs):
        if epoch % batches_per_epoch == 0:
            # Reinitialising Dataloader
            iter_monet = iter(dataloader_monet)
            iter_photo = iter(dataloader_photo)

        images_monet, _ = iter_monet.next()
        images_photo, _ = iter_photo.next()
        if train_on_gpu:
            images_monet = images_monet.cuda()
            images_photo = images_photo.cuda()

        # discriminator training

        d_x_optimizer.zero_grad()
        out_x = D_X(images_monet)
        d_x_real_loss = loss.real_mse_loss(out_x)

        fake_x = G_YtoX(images_photo)
        out_x_fake = D_X(fake_x)
        d_x_fake_loss = loss.fake_mse_loss(out_x_fake)

        d_x_loss = d_x_real_loss + d_x_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()


        d_y_optimizer.zero_grad()
        out_y = D_Y(images_photo)
        d_y_real_loss = loss.real_mse_loss(out_y)

        fake_y = G_XtoY(images_monet)
        out_x_fake = D_X(fake_y)
        d_y_fake_loss = loss.fake_mse_loss(out_x_fake)

        d_y_loss = d_y_real_loss + d_y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()

        # generator training
        g_optimizer.zero_grad()

        fake_x = G_YtoX(images_photo)
        out_x_fake = D_X(fake_x)
        g_x_loss = loss.real_mse_loss(out_x_fake)

        fake_y =G_XtoY(images_monet)
        out_y_fake = D_Y(fake_y)
        g_y_loss = loss.real_mse_loss(out_y_fake)

        reconstructed_Y = G_XtoY(fake_x)
        reconstructed_y_loss = loss.cycle_consistency_loss(images_photo, reconstructed_Y,10)

        reconstructed_X = G_YtoX(fake_x)
        reconstructed_x_loss = loss.cycle_consistency_loss(images_monet, reconstructed_X, 10)

        g_total_loss = g_y_loss + g_x_loss + reconstructed_y_loss + reconstructed_x_loss
        g_total_loss.backward()
        g_optimizer.step()


        if epoch % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))





    
