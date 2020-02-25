# from models.stargan.model import Generator, Discriminator
import importlib
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from sys import exit
import tensorboardX as tbx

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, rafd_loader, config):
        """Initialize configurations."""
        self.config = config
        # Data loader.
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_sty = config.lambda_sty
        self.lambda_ds = config.lambda_ds
        self.default_lambda_ds = config.lambda_ds
        self.lambda_cyc = config.lambda_cyc
        self.lambda_gp = config.lambda_gp
        self.latent_code_dim = config.latent_code_dim
        self.style_code_dim = config.style_code_dim

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.f_lr = config.f_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        self.writer = tbx.SummaryWriter(self.log_dir)


    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = getattr(importlib.import_module(self.config.network_G), 'Generator')(in_dim=3, image_size=self.image_size, style_dim=self.style_code_dim)
        self.D = getattr(importlib.import_module(self.config.network_D), 'Discriminator')(in_channel=3, image_size=self.image_size, num_domain=self.c_dim, D=1, max_dim=1024)
        self.mapping_function = getattr(importlib.import_module(self.config.network_D), 'MappingNetwork')(in_dim=self.latent_code_dim, style_dim=self.style_code_dim, hidden_dim=512, num_domain=self.c_dim, num_layers=6, pixel_norm=False)
        self.style_encoder = getattr(importlib.import_module(self.config.network_D), 'StyleEncoder')(in_channel=3, image_size=self.image_size, num_domain=self.c_dim, D=64, max_dim=512)
        
        # Initialize weights
        self.G.apply(self.weight_init_kaiming_normal)
        self.D.apply(self.weight_init_kaiming_normal)
        self.mapping_function.apply(self.weight_init_kaiming_normal)
        self.style_encoder.apply(self.weight_init_kaiming_normal)

        # Exponential moving averages
        self.G_running = getattr(importlib.import_module(self.config.network_G), 'Generator')(in_dim=3, image_size=self.image_size, style_dim=self.style_code_dim)
        self.mapping_function_running = getattr(importlib.import_module(self.config.network_D), 'MappingNetwork')(in_dim=self.latent_code_dim, style_dim=self.style_code_dim, hidden_dim=512, num_domain=self.c_dim, num_layers=6, pixel_norm=False)
        self.style_encoder_running = getattr(importlib.import_module(self.config.network_D), 'StyleEncoder')(in_channel=3, image_size=self.image_size, num_domain=self.c_dim, D=64, max_dim=512)
        
        self.G_running.train(False)
        self.mapping_function_running.train(False)
        self.style_encoder_running.train(False)

        self.accumulate(self.G_running, self.G, 0)
        self.accumulate(self.mapping_function_running, self.mapping_function, 0)
        self.accumulate(self.style_encoder_running, self.style_encoder, 0)

        # self.g_optimizer = torch.optim.Adam([
        #         {'params': self.G.parameters(), 'lr':self.g_lr, 'betas': (self.beta1, self.beta2)},
        #         {'params': self.mapping_function.parameters(), 'lr': self.f_lr, 'betas': (self.beta1, self.beta2)},
        #         {'params': self.style_encoder.parameters(), 'lr':self.g_lr, 'betas': (self.beta1, self.beta2)},
        #     ])
        # self.g_optimizer = torch.optim.Adam([
        #         {'params': self.G.parameters(), 'lr':self.g_lr, 'betas': (self.beta1, self.beta2)},
        #         {'params': self.mapping_function.parameters(), 'lr': self.f_lr, 'betas': (self.beta1, self.beta2)},
        #         {'params': self.style_encoder.parameters(), 'lr':self.g_lr, 'betas': (self.beta1, self.beta2)},
        #     ])
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.mf_optimizer = torch.optim.Adam(self.mapping_function.parameters(), self.f_lr, [self.beta1, self.beta2])
        self.enc_optimizer = torch.optim.Adam(self.style_encoder.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        self.G.to(self.device)
        self.D.to(self.device)
        self.mapping_function.to(self.device)
        self.style_encoder.to(self.device)

        self.G_running.to(self.device)
        self.mapping_function_running.to(self.device)
        self.style_encoder_running.to(self.device)

    def accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.mf_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def create_labels(self, c_org, c_dim=5, dataset='CelebA'):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    
    def weight_init_kaiming_normal(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
            
            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            # To one-hot
            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)
            
            # Set tensor to cuda
            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            # print("label_org:", label_org) -> label_org: tensor([0, 2, 1, 0, 2, 2, 1, 2])
            x_code_1 = torch.randn(self.batch_size, self.latent_code_dim).to(self.device)
            x_code_2 = torch.randn(self.batch_size, self.latent_code_dim).to(self.device)
            x_code_3 = torch.randn(self.batch_size, self.latent_code_dim).to(self.device)

            # Adversarial ground truths
            self.adv_loss = torch.nn.MSELoss().cuda()
            lsgan_true = torch.Tensor(self.batch_size, 1).fill_(1.0).to(self.device)
            lsgan_fake = torch.Tensor(self.batch_size, 1).fill_(0.0).to(self.device)

            # =================================================================================== #
            #                             2. Forward the parameters                               #
            # =================================================================================== #

            # Compute style code
            x_style_code_1 = self.mapping_function(x_code_1, label_trg) # [self.batch_size, self.style_dim]
            x_style_code_2 = self.mapping_function(x_code_2, label_trg)
            x_style_code_3 = self.mapping_function(x_code_3, label_trg)

            # Forward generator
            x_fake_1 = self.G(x_real, x_style_code_1) # For caliculating the adv loss (Eq.1)
            x_fake_2 = self.G(x_real, x_style_code_2) # For caliculating the ds loss (Eq.3)
            x_fake_3 = self.G(x_real, x_style_code_3) # For caliculating the ds loss (Eq.3)

            # Forward reconstruction
            ## Real -> for caliculating the cyc rec loss (Eq.4)
            x_real_style_code = self.style_encoder(x_real, label_org) # [bs, style_dim]
            x_fake_rec = self.G(x_fake_1, x_real_style_code)
            ## Fake -> for caliculating the sty rec loss (Eq.2)
            x_fake_style_code = self.style_encoder(x_fake_1, label_trg) # [bs, style_dim]

            # =================================================================================== #
            #                             3. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src = self.D(x_real, label_org)
            # d_loss_real = - torch.mean(out_src) # WGAN
            d_loss_real = self.adv_loss(out_src, lsgan_true)

            # Compute loss with fake images.
            out_src = self.D(x_fake_1.detach(), label_trg)
            # d_loss_fake = torch.mean(out_src) # WGAN
            d_loss_fake = self.adv_loss(out_src, lsgan_fake)

            # # Compute loss for gradient penalty.
            # alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            # x_hat = (alpha * x_real.data + (1 - alpha) * x_fake_1.data).requires_grad_(True)
            # out_src = self.D(x_hat, label_org)
            # d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = 0.5 * (d_loss_real + d_loss_fake) # + self.lambda_gp * d_loss_gp
            # d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            # loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               4. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain. (Eq.1)
                out_src = self.D(x_fake_1, label_trg) # TODO: Maybe `label_trg` is ok.

                # g_loss_fake = - torch.mean(out_src) # WGAN
                g_loss_fake = self.adv_loss(out_src, lsgan_true)
                
                # Style reconstruction. (Eq. 2)
                g_loss_sty = torch.mean(torch.abs(x_style_code_1 - x_fake_style_code))

                # Style diversification. (Eq.3)
                g_loss_ds = torch.mean(torch.abs(x_fake_2 - x_fake_3))

                # Target-to-original domain. (Eq.4)
                g_loss_rec = torch.mean(torch.abs(x_real - x_fake_rec))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_sty * g_loss_sty - self.lambda_ds * g_loss_ds + self.lambda_cyc * g_loss_rec
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                self.mf_optimizer.step()
                self.enc_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_sty'] = g_loss_sty.item()
                loss['G/loss_ds'] = g_loss_ds.item()

                self.accumulate(self.G_running, self.G)
                self.accumulate(self.mapping_function_running, self.mapping_function)
                self.accumulate(self.style_encoder_running, self.style_encoder)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                # if self.use_tensorboard:
                #     for tag, value in loss.items():
                #         self.logger.scalar_summary(tag, value, i+1)
                for tag, value in loss.items():
                    self.writer.add_scalar(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for _ in range(4):
                        x_code = torch.randn(self.batch_size, self.latent_code_dim).to(self.device)
                        x_style_code = self.mapping_function_running(x_code, label_trg)
                        x_fake_list.append(self.G_running(x_fixed, x_style_code))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                MF_path = os.path.join(self.model_save_dir, '{}-MF.ckpt'.format(i+1))
                Enc_path = os.path.join(self.model_save_dir, '{}-Enc.ckpt'.format(i+1))

                G_running_path = os.path.join(self.model_save_dir, '{}-G_running.ckpt'.format(i+1))
                MF_running_path = os.path.join(self.model_save_dir, '{}-MF_running.ckpt'.format(i+1))
                Enc_running_path = os.path.join(self.model_save_dir, '{}-Enc_running.ckpt'.format(i+1))


                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.G.state_dict(), MF_path)
                torch.save(self.D.state_dict(), Enc_path)

                torch.save(self.G_running.state_dict(), G_running_path)
                torch.save(self.mapping_function_running.state_dict(), MF_running_path)
                torch.save(self.style_encoder_running.state_dict(), Enc_running_path)

                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # # Decay learning rates.
            # if i % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
            #     g_lr -= (self.g_lr / float(self.num_iters_decay))
            #     d_lr -= (self.d_lr / float(self.num_iters_decay))
            #     self.update_lr(g_lr, d_lr)
            #     print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            # Decay the weight of lambda_ds.
            if (i+1) > self.num_iters_decay:
                self.lambda_ds -= float(self.default_lambda_ds / self.num_iters_decay)

        self.writer.close()

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))