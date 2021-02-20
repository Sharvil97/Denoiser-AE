import os
import model_dcgan
import pytorch_ssim 
import numpy as np
from utils_dcgan import weights_init, weights_init_kaiming
import torch.optim as optim
import torchvision.utils as vutils
import losses as L

import torch
import torch.nn as nn 
from torch.autograd import Variable

from azureml.core.run import Run
from azureml.core import Dataset, Run
from azureml.core import Workspace, Dataset

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data import DirtyDocumentsDataset, Rescale, ToTensor, RandomCrop, ImgAugTransform
import torchvision.transforms as transforms


def main(batch_size, lr, img_size, epochs, num_layers):
    batch_size=batch_size
    img_size = img_size
    learning_rate=lr 
    # weight_decay=1e-5
    epochs=epochs
    
    run = Run.get_context()
    workspace = run.experiment.workspace

    dataset_name = 'datadenoisingnosiy'
    dataset_name2 = 'datadenoisyclean'

    # Get a dataset by name
    daekaggle_trainclean = Dataset.get_by_name(workspace=workspace, name=dataset_name2)
    daekaggle_trainnoisy = Dataset.get_by_name(workspace=workspace, name=dataset_name)

    daekaggle_trainclean.download(target_path='./train_clean', overwrite=True)
    daekaggle_trainnoisy.download(target_path='./train_noisy', overwrite=True)    

    #Dataloader
    composed = transforms.Compose([ImgAugTransform(), RandomCrop((256,256),0.25),Rescale((img_size, img_size)),ToTensor()])

    data_folder_noisy = "./train_noisy/"
    data_folder_clean = "./train_clean/"
    train_dataset = DirtyDocumentsDataset(dirty_dir=data_folder_noisy, clean_dir=data_folder_clean, transform=composed)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

    dncnn=model_dcgan.Generator(num_layers).cuda(0)
    dncnn=nn.DataParallel(dncnn)
    dncnn.apply(weights_init_kaiming)    

    discriminator = model_dcgan.Discriminator().cuda(0)
    discriminator=nn.DataParallel(discriminator)
    discriminator.apply(weights_init)    


    #losses

    ssim=pytorch_ssim.SSIM()
    # bce_loss=nn.BCELoss()
    mse_loss=nn.MSELoss()
    gen_criterion=L.GenLoss("hinge",False)
    dis_criterion=L.DisLoss("hinge",False)

    optimizerG=optim.Adam(dncnn.parameters(),lr=learning_rate)
    optimizerD=optim.Adam(discriminator.parameters(),lr=3*learning_rate)

    os.makedirs('./outputs/models', exist_ok=True)
    os.makedirs("./outputs/images/real_Dirty", exist_ok=True)
    os.makedirs("./outputs/images/real_Clean", exist_ok=True)
    os.makedirs("./outputs/images/fake_Clean", exist_ok=True)

    for epoch in range(epochs):


        if epoch>=20 and epoch %3==0:
            optimizerD.param_groups[0]['lr'] *= 0.9
            optimizerG.param_groups[0]['lr'] *= 0.9

        for i,data in enumerate(train_loader):
            train_image = data['train_image']
            clean_image = data['clean_image']

            train_image.cuda(0)
            clean_image.cuda(0)


            dncnn.zero_grad()


            denoised=dncnn(train_image)
            ssim0=ssim(denoised,clean_image).item()
            ssim1=ssim(train_image,clean_image).item()

    #train G
            cls_fake=discriminator(denoised)
            errG=gen_criterion(cls_fake)


            mse=mse_loss(denoised,clean_image)
            loss=mse*100+errG
            loss.backward()

            optimizerG.step()


    #train D

            if i%1==0:
                discriminator.zero_grad()
                fake=denoised.detach()
                cls_fake=discriminator(fake)
                cls_real=discriminator(clean_image)
                errors=dis_criterion(cls_fake,cls_real)



                errors.backward()
                optimizerD.step()



            if i%1==0:
                print('epoch:%d batch:%d || ssim: %.4f ~ %.4f || loss: %.4f || dis: %.4f || mse: %.4f ' % (
                epoch, i + 1, ssim0, ssim1, loss.item(), errG, mse.item()))

                run.log("Generator Loss Iteration", np.float(loss.item()))
                run.log("Discriminator Loss Iteration", np.float(errors.item()))
                run.log("MSE Loss Iteration", np.float(mse.item()))
                run.log("SSIM 0 Loss Iteration", np.float(ssim0))
                run.log("SSIM 1 Loss Iteration", np.float(ssim1))



        if  epoch%1==0:
            torch.save(dncnn.state_dict(), 'outputs/models/dncnn.pth')
            torch.save(discriminator.state_dict(), 'outputs/models/discriminator.pth')


        save_image(train_image[:,:,:],'./outputs/images/real_Dirty/{}.png'.format(epoch+1))
        save_image(clean_image[:,:,:],'./outputs/images/real_Clean/{}.png'.format(epoch+1))
        save_image(denoised[:,:,:],'./outputs/images/fake_Clean/{}.png'.format(epoch+1))

        run.log("Generator Loss", np.float(loss.item()))
        run.log("Discriminator Loss", np.float(errors.item()))
        run.log("MSE Loss", np.float(mse.item()))
        run.log("SSIM 0 Loss", np.float(ssim0))
        run.log("SSIM 1 Loss", np.float(ssim1))

if __name__ == "__main__":
    batch_size=8
    lr = 2e-5
    img_size=512
    epochs=1000
    num_layers = 16

    main(batch_size, lr, img_size, epochs, num_layers)