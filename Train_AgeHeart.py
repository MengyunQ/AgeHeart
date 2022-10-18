import torch
from torch import optim
from torch.utils.data import DataLoader
from utils import *
from dataset import *
import networks
from image_utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
# from matplotlib.fontmanager import FontProperties
import os
from torch.utils.tensorboard import SummaryWriter
import copy
import argparse
import numpy as np
import torch.nn.functional as F
import random
from textwrap import wrap
import faulthandler
# from visualize import *
from loss import *
from torch.nn.functional import l1_loss
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
from calculate_clinical_features import *
from Metric_utils import Dice_of_sequence
from collections import defaultdict
import consts
from tqdm import tqdm

faulthandler.enable()

def ReconLoss(recon_x, x, args = None):
    classweight = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)
    x_time = torch.argmax(x, axis=1)
    CE = F.cross_entropy(recon_x, x_time, reduction='mean', weight=classweight)

    return CE

def train(epoch, args=None):
    # train model
    E.train()
    Dz.train()
    Dimg.train()
    G.train()
    Age_classifier.train()
    losses = defaultdict(lambda: [])
    train_loss = 0

    for batch_idx, (images, age, age_target, sex, segID) in tqdm(enumerate(train_dataloader)):
        images = images.to(device, dtype=torch.float32)
        age = age.to(device)
        age_target = age_target.to(device)
        if args.noise:
            labels = add_noise_to_label(age, sex)
            targets = add_noise_to_label(age_target,sex)
        else:
            labels = torch.stack([str_to_tensor(a, s, normalize=True) for i, (a, s) in enumerate(zip(age, sex), 1)])
            targets = torch.stack([str_to_tensor(a, s, normalize=True) for i, (a, s) in enumerate(zip(age_target, sex), 1)])
        labels = labels.to(device=device)
        targets = targets.to(device=device)

        z = E(images)
        # Input\Output Loss
        z_l = torch.cat((z.detach(), labels), 1)
        generated = G(z_l, args)
        eg_loss = input_output_loss(generated, images)
        losses['eg'].append(eg_loss.item())

        # DiscriminatorZ Loss
        z_prior = two_sided(torch.rand_like(z, device=device))  # [-1 : 1]
        d_z_prior = Dz(z_prior.detach())
        d_z = Dz(z.detach())
        dz_loss_prior = bce_with_logits_loss(d_z_prior, torch.ones_like(d_z_prior))
        dz_loss = bce_with_logits_loss(d_z, torch.zeros_like(d_z))
        dz_loss_tot = (dz_loss + dz_loss_prior)
        losses['dz'].append(dz_loss_tot.item())

        # Encoder\DiscriminatorZ Loss
        ez_loss = 0.0001 * bce_with_logits_loss(d_z, torch.ones_like(d_z))
        ez_loss.to(device)
        losses['ez'].append(ez_loss.item())

        # DiscriminatorImg Loss
        d_i_input = Dimg(images.detach(), labels.detach(), device)
        d_i_output = Dimg(generated.detach(), labels.detach(), device)

        di_input_loss = bce_with_logits_loss(d_i_input, torch.ones_like(d_i_input))
        di_output_loss = bce_with_logits_loss(d_i_output, torch.zeros_like(d_i_output))
        di_loss_tot = (di_input_loss + di_output_loss)
        losses['di'].append(di_loss_tot.item())

        # Generator\DiscriminatorImg Loss
        dg_loss = 0.0001 * bce_with_logits_loss(d_i_output, torch.ones_like(d_i_output))
        losses['dg'].append(dg_loss.item())

        ##add the age loss
        age_logit =Age_classifier(generated)
        age_loss = F.cross_entropy(age_logit, age.long())
        losses['age'].append(age_loss.item())
        ## cycle loss
        z_l = torch.cat((z.detach(), targets), 1)
        generated_t = G(z_l, args)

        if args.dit_loss_weight != 0:
            age_pre_target = Age_classifier(generated_t)
            target_age_loss = F.cross_entropy(age_pre_target, age_target.long())
            losses['age_target'].append(target_age_loss.item())
        else:
            target_age_loss = 0

        if args.dit_loss_weight != 0:
            # target img loss
            d_i_input_t = Dimg(images.detach(), targets.detach(), device)
            d_i_output_t = Dimg(generated_t.detach(), targets.detach(), device)

            di_input_loss_t = bce_with_logits_loss(d_i_input_t, torch.ones_like(d_i_input_t))
            di_output_loss_t = bce_with_logits_loss(d_i_output_t, torch.zeros_like(d_i_output_t))
            di_loss_tot_t = (di_input_loss_t + di_output_loss_t)
            losses['di_t'].append(di_loss_tot_t.item())
        else:
            di_loss_tot_t = 0
        if args.cyc_loss_weight != 0:
                ## cyc_loss
            z_c = E(generated_t)
            z_c = torch.cat((z_c.detach(), labels), 1)
            generated_cyc = G(z_c, args)
            cyc_loss = input_output_loss(generated_cyc, images)
            losses['eg_cyc'].append(cyc_loss.item())
            # target img loss
        else:
            cyc_loss = 0

        di_loss_tot = di_loss_tot + di_loss_tot_t * float(args.dit_loss_weight)


        loss = eg_loss + ez_loss + dg_loss \
               + float(args.age_loss_weight) * age_loss \
               + float(args.age_loss_weight) * target_age_loss \
               + float(args.cyc_loss_weight) * cyc_loss\


        # Back prop on Encoder\Generator
        eg_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        eg_optimizer.step()

        # Back prop on DiscriminatorZ
        dz_optimizer.zero_grad()
        dz_loss_tot.backward(retain_graph=True)
        dz_optimizer.step()

        # Back prop on DiscriminatorImg
        di_optimizer.zero_grad()
        di_loss_tot.backward()
        di_optimizer.step()
        train_loss += loss.item()
    lossmean = {}
    lossmean.update({f'{key}': [] for key in losses.keys()})
    for key in losses.keys():
        lossmean[key]=np.mean(np.array(losses[key]))

    train_loss /= batch_idx + 1
    # Dis_loss /= batch_idx + 1
    print('====> Epoch: {} Average loss: Train {:.4f} '.format(epoch, train_loss))
    return lossmean, train_loss


def test(epoch):
    E.eval()
    G.eval()
    val_loss=0
    dice_ed = 0
    dice_es = 0


    # dice_sample= []
    # dice_recon= []

    with torch.no_grad():
        for batch_idx, (images, age, age_target, sex, segID) in tqdm(enumerate(valid_dataloader)):
            images = images.to(device)
            if args.noise:
                labels = add_noise_to_label(age, sex)
            else:
                labels = torch.stack([str_to_tensor(a, s, normalize=True) for i, (a, s) in enumerate(zip(age, sex), 1)])

            validate_labels = labels.to(device)

            z = E(images)
            z_l = torch.cat((z, validate_labels), 1)
            recon_batch = G(z_l,args)

            valloss = input_output_loss(images, recon_batch)
            val_loss += valloss.item()
            ## visual
            n = min(images.size(0), 8)
            plt.style.use('default')
            fig, axs = plt.subplots(nrows=4, ncols=n, frameon=True,
                                    gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
            fig.set_size_inches([n*3, 4*3])
            [ax.set_axis_off() for ax in axs.ravel()]
            val_dice_each_ED = []
            val_dice_each_ES = []

            for k in range(0, n):
                seg_vol_ED = onehot2label(images[k, 0:4].cpu().detach().numpy())
                seg_vol_ES = onehot2label(images[k, 4:8].cpu().detach().numpy())
                recon_ED = onehot2label(recon_batch[k, 0:4].cpu().detach().numpy())
                recon_ES = onehot2label(recon_batch[k, 4:8].cpu().detach().numpy())
                dsc_ED = np.mean(np_mean_dice(recon_ED, seg_vol_ED))
                dsc_ES = np.mean(np_mean_dice(recon_ES, seg_vol_ES))
                val_dice_each_ED.append(dsc_ES)
                val_dice_each_ES.append(dsc_ES)

                view1 = seg_vol_ED[int(seg_vol_ED.shape[0]/2), :, :]
                recon_view1 = recon_ED[int(seg_vol_ED.shape[0]/2), :, :]
                view2 = seg_vol_ES[int(seg_vol_ED.shape[0]/2), :, :]
                recon_view2 = recon_ES[int(seg_vol_ED.shape[0]/2), :, :]
                if n==1:
                    axs[0].imshow(view1, clim=(0, 4))
                    axs[1].imshow(recon_view1, clim=(0, 4))
                    axs[2].imshow(view2, clim=(0, 4))
                    axs[3].imshow(recon_view2, clim=(0, 4))

                    axs[0].set_title('{:0.2f}'.format(dsc_ED))
                    axs[2].set_title('{:0.2f}'.format(dsc_ES))

                else:
                    axs[0, k].imshow(view1, clim=(0, 4))
                    axs[1, k].imshow(recon_view1, clim=(0, 4))
                    axs[2, k].imshow(view2, clim=(0, 4))
                    axs[3, k].imshow(recon_view2, clim=(0, 4))

                    axs[0, k].set_title('{:0.2f}'.format(dsc_ED))
                    axs[2, k].set_title('{:0.2f}'.format(dsc_ES))
            dice_ed += np.mean(np.array(val_dice_each_ED))
            dice_es += np.mean(np.array(val_dice_each_ES))

            writer.add_figure('Test', fig, epoch)

        test_single(E, G, images[0], age[0], sex[0])
            # loss update and log
    val_loss /= batch_idx + 1
    dice_ed /= batch_idx + 1
    dice_es /= batch_idx + 1
    # Dice_sample=np.mean(np.array(dice_sample))
    print('====> Epoch: {} Average loss: Valid {:.4f} '.format(epoch, val_loss))

    return val_loss, dice_ed, dice_es

def test_single(E, G, image_tensor, age, gender):

    batch = image_tensor.repeat(consts.NUM_AGES, 1, 1, 1, 1).to(device=device)  # N x D x H x W
    z = E(batch)
    if args.noise:
        l = generate_agegroups_label(gender).to(device)# N x Z
    else:
        gender_tensor = -torch.ones(consts.NUM_GENDERS)
        gender_tensor[int(gender)] *= -1
        gender_tensor = gender_tensor.repeat(consts.NUM_AGES, consts.NUM_AGES // consts.NUM_GENDERS)  # apply gender on all images

        age_tensor = -torch.ones(consts.NUM_AGES, consts.NUM_AGES)
        for i in range(consts.NUM_AGES):
            age_tensor[i][i] *= -1  # apply the i'th age group on the i'th image

        l = torch.cat((age_tensor, gender_tensor), 1).to(device)
    z_l = torch.cat((z, l), 1)

    generated = G(z_l, args)
    plt.style.use('default')
    fig, axs = plt.subplots(nrows=4, ncols=consts.NUM_AGES, frameon=True,
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    fig.set_size_inches([14, consts.NUM_AGES])
    [ax.set_axis_off() for ax in axs.ravel()]

    for k in range(0, consts.NUM_AGES):
        reconED = onehot2label(generated[k, 0:4].cpu().detach().numpy())
        reconES = onehot2label(generated[k, 4:8].cpu().detach().numpy())
        recon_viewED = reconED[int(reconED.shape[0] / 2), :, :]
        recon_viewES = reconES[int(reconES.shape[0] / 2), :, :]

        if k == 0:
            errorED = np.zeros((128,128))
            errorES = np.zeros((128,128))
            vol_ED_first = recon_viewED
            vol_ES_first = recon_viewES
        else:
            errorED = recon_viewED - vol_ED_first
            errorES = recon_viewES - vol_ES_first

        axs[0, k].imshow(recon_viewED, clim=(0, 4))
        ## put error map between recon and seg1, seg2
        axs[2, k].imshow(recon_viewES, clim=(0, 4))

        axs[1, k].imshow(errorED, clim=(-8, 8), cmap='seismic')
        axs[3, k].imshow(errorES, clim=(-8, 8), cmap='seismic')

        axs[0, k].text(5, 15, f'Age{45 + 5 * k}-{50 + 5 * k}', fontsize=10, c='white')

    writer.add_figure('Aging', fig, epoch)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='C-VAE-4D')
    parser.add_argument('-b','--batch_size', default=32) ## for 24G
    parser.add_argument('--lr', default=2e-4)
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--weight_decay', default=1e-5)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--age_loss_weight', default=0.01)
    parser.add_argument('--cyc_loss_weight', default=0)
    parser.add_argument('--dit_loss_weight', default=0)
    parser.add_argument('--loss', default='l1',choices=['l1','bce'])
    parser.add_argument('--noise', action='store_true', help='add noise in age code')



    args = parser.parse_args()

    # set visible GPU env
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # hyperparameter
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    N = int(args.epochs)
    weight_decay = float(args.weight_decay)
    model_type = f'caae_40k_sampled_loss_age{args.age_loss_weight}' \
                     f'_zdim{consts.NUM_Z_CHANNELS}_style_{consts.NUM_STYLE}' \
                     f'_cyc_{args.cyc_loss_weight}_dit_{args.dit_loss_weight}_codemapping_{args.loss}'


    if args.noise:
        model_type = f'{model_type}_noiseage'


    # build model
    E = networks.Encoder_caae()
    Dz = networks.DiscriminatorZ_caae()
    Dimg = networks.DiscriminatorImg_caae()
    G = networks.Generator_caae(args)
    Age_classifier = networks.AgeClassifier()


    # loss function and optimizer

    eg_optimizer = optim.Adam(list(E.parameters())+ list(Age_classifier.parameters()) + list(G.parameters()),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    dz_optimizer = optim.Adam(Dz.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    di_optimizer = optim.Adam(Dimg.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)


    E.to(device)
    Dz.to(device)
    Dimg.to(device)
    G.to(device)
    Age_classifier.to(device)
    if args.loss == 'l1':
        input_output_loss = l1_loss
    elif args.loss == 'bce':
        input_output_loss = bce_with_logits_loss

    best_model, best_loss = [], 1e10

    # load dataset
    save_path = './Results_AgeHeart'
    dest_dir = './UKbiobank/'
    txt_path = f'{dest_dir}label/uk40k_edes_select_train.csv'
    train_dataset = UKbiobank_40k_EDES_cycle(dest_dir, txt_path)
    txt_path = f'{dest_dir}label/uk40k_edes_select_test.csv'
    val_dataset = UKbiobank_40k_EDES_cycle(dest_dir, txt_path)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    valid_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)

    model_name = f'{model_type}_batch_{batch_size}_epochs_{N}_modifyloss.pt'
    print(model_name)
    folder = 'caae'
    logdir = os.path.join(f'{save_path}/log/{folder}/', model_name[0:-3])
    writer = SummaryWriter(logdir)
    writer.add_hparams({'type': model_type, 'epochs': N}, {})

    # train
    for epoch in tqdm(range(0, N)):
        train_loss, train_loss_all = train(epoch, args=args)
        # update best loss
        modelpath = f'{save_path}/models/{folder}/{model_name[0:-3]}/'
        setup_dir(modelpath)
        if train_loss_all< best_loss:
            best_loss = train_loss_all
            torch.save(E.state_dict(), os.path.join(modelpath, 'bestloss_E.pt'))
            torch.save(G.state_dict(), os.path.join(modelpath, 'bestloss_G.pt'))
        if (epoch>0) & (epoch % 25 == 0):
            torch.save(E.state_dict(), os.path.join(modelpath, f'epoch{epoch}_E.pt'))
            torch.save(G.state_dict(), os.path.join(modelpath, f'epoch{epoch}_G.pt'))
        #sample_latent_motion(epoch, args, 2)

        val_loss,dice_ed,dice_es = test(epoch)
        # #shot_latent(epoch, test_mu, test_logvar)

        writer.add_scalars('Train_loss', train_loss, epoch)
        writer.add_scalars('Val_Loss', {'Val': val_loss}, epoch)
        writer.add_scalars('Test_Dice', {'ED': dice_ed, 'ES':dice_es}, epoch)

    writer.close()
