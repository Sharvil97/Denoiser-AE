import itertools

from model_cyclegan import Generator
from model_cyclegan import Discriminator

from utils import ReplayBuffer
from utils import LambdaLR 
from utils import weights_init_normal

import torch
from torch.autograd import Variable
from torch.nn import DataParallel

from azureml.core.run import Run
from azureml.core import Dataset, Run
from azureml.core import Workspace, Dataset

from data import DirtyDocumentsDataset

def main(input_channels, output_channels, lr, num_epochs, offset_epochs, decay_start_epoch, batch_size, output_size_w, output_size_h):

    run = Run.get_context()
    workspace = run.experiment.workspace

    dataset_name = 'datadenoisingnosiy'
    dataset_name2 = 'datadenoisyclean'

    # Get a dataset by name
    daekaggle_trainclean = Dataset.get_by_name(workspace=workspace, name=dataset_name2)
    daekaggle_trainnoisy = Dataset.get_by_name(workspace=workspace, name=dataset_name)

    daekaggle_trainclean.download(target_path='./train_clean', overwrite=True)
    daekaggle_trainnoisy.download(target_path='./train_noisy', overwrite=True)    

    #define networks
    G_A2B = Generator(input_channels, output_channels).cuda(0)
    # parallelize
    G_A2B = nn.DataParallel(G_A2B)

    G_B2A = Generator(output_channels, input_channels).cuda(0)
    #parallelize
    G_B2A = nn.DataParallel(G_B2A) 

    D_A = Discriminator(input_channels).cuda(0)
    #parallelize
    D_A = nn.DataParallel(D_A) 

    D_B = Discriminator(output_channels).cuda(0)
    #parallelize
    D_B = nn.DataParallel(D_B) 

    G_A2B.apply(weights_init_normal())
    G_B2A.apply(weights_init_normal())
    D_A.apply(weights_init_normal())
    D_B.apply(weights_init_normal())

    #define loss
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    #optimizers and LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
                                    lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(num_epochs, offset_epochs, decay_start_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(num_epochs, offset_epochs, decay_start_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(num_epochs, offset_epochs, decay_start_epoch).step)

    #inputs
    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(batch_size, input_channels, output_size_w, output_size_h)
    input_B = Tensor(batch_size, input_channels, output_size_w, output_size_h)
    target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    #Dataloader
    composed = transforms.Compose([ImgAugTransform(), RandomCrop((256,256),0.25),Rescale((615, 799)),ToTensor()])

    data_folder_noisy = "./train_noisy/"
    data_folder_clean = "./train_clean/"
    train_dataset = DirtyDocumentsDataset(dirty_dir=data_folder_noisy, clean_dir=data_folder_clean, transform=composed)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

    os.makedirs('./outputs/models', exist_ok=True)
    os.makedirs("./outputs/images/real_Dirty", exist_ok=True)
    os.makedirs("./outputs/images/real_Clean", exist_ok=True)
    os.makedirs("./outputs/images/fake_Clean", exist_ok=True)
    os.makedirs("./outputs/images/fake_Dirty", exist_ok=True)

    #training

    for epochs in range(num_epochs):
        for i, batch in enumerate(train_loader):
            #model input
            real_A = Variable(input_A.copy_(batch['train_noisy'])).cuda(0)
            real_B = Variable(input_B.copy_(batch['train_clean'])).cuda(0)

            #Generators
            optimizer_G.zero_grad()


            # Identity Loss
            # G_A2B(B) should be equal to B if real B is fed
            same_B = G_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0

            # G_B2A(A) should be equal to A if real A is fed
            same_A = G_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN Loss

            fake_B = G_A2B(real_A)
            pred_fake = D_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = G_B2A(real_A)
            pred_fake = D_A(fake_B)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
            
            # Cycle loss
            recovered_A = G_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = G_A2B(fake_A)
            loss_cycle_B2B = criterion_cycle(recovered_B, real_B)*10.0

            #Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A\
                + loss_cycle_ABA + loss_cycle_B2B
            loss_G.backward()

            optimizer_G.step()

            # Discriminator A
            optimizer_D_A.zero_grad()

            #Real Loss
            pred_real = D_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            #fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = D_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            #Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            # Discriminator B
            optimizer_D_B.zero_grad()

            #Real Loss
            pred_real = D_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            #Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = D_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            #total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()

            # Logg and print loss per step
            print(f"Step: {i+1}")
            print(f"loss_G: {loss_G}")
            print(f"loss_G_identity: {(loss_identity_A + loss_identity_B)}")
            print(f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A)}")
            print(f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB)}")
            print(f"loss_D: {(loss_D_A + loss_D_B)}")

        # Logg and print loss
        print(f"Epoch: {epochs+1}")
        print(f"loss_G: {loss_G}")
        print(f"loss_G_identity: {(loss_identity_A + loss_identity_B)}")
        print(f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A)}")
        print(f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB)}")
        print(f"loss_D: {(loss_D_A + loss_D_B)}")
        
        run.log("loss_G", loss_G)
        run.log("loss_G_identity", (loss_identity_A + loss_identity_B))
        run.log("loss_G_GAN", (loss_GAN_A2B + loss_GAN_B2A))
        run.log("loss_G_cycle", (loss_cycle_ABA + loss_cycle_BAB))
        run.log("loss_D", (loss_D_A + loss_D_B))    

        save_image(real_A[:,:,:],'./outputs/images/real_Dirty/{}.png'.format(epochs+1))
        save_image(real_B[:,:,:],'./outputs/images/real_Clean/{}.png'.format(epochs+1))
        save_image(fake_A[:,:,:],'./outputs/images/fake_Clean/{}.png'.format(epochs+1))
        save_image(fake_B[:,:,:],'./outputs/images/fake_Dirty/{}.png'.format(epochs+1))
   


        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()    
            
        
        # Save models checkpoints
        torch.save(G_A2B.state_dict(), 'outputs/models/G_A2B.pth')
        torch.save(G_B2A.state_dict(), 'outputs/models/G_B2A.pth')
        torch.save(D_A.state_dict(), 'outputs/models/D_A.pth')
        torch.save(D_B.state_dict(), 'outputs/models/D_B.pth')

        save_image(train_image[:,:,:],'./outputs/v1.1_input/{}.png'.format(i+1))


if __name__ == "__main__":
    input_channels = 1
    output_channels = 1
    lr = 2e-5
    num_epochs = 1000
    offset_epochs = 
    decay_start_epoch = 
    batch_size = 128
    output_size_w = 615
    output_size_h = 799

    main(input_channels, output_channels, lr, num_epochs, offset_epochs, decay_start_epoch, batch_size, output_size_w, output_size_h)





