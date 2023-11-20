import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch import autograd
from einops import rearrange

from vae import VarAutoEncoder, MLPVAE
import pytorch_fid_wrapper as pfw

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--save_every', type=int, default=5)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--output_results', type=str, default='attack_results_mlp')
parser.add_argument('--output_model', type=str, default='output_model_mlp')
parser.add_argument('--batch_size_train', type=int, default=128)
parser.add_argument('--batch_size_test', type=int, default=128)
parser.add_argument('--kl_beta', type=float, default=1.0)
parser.add_argument('--iters', type=int, default=1)
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id>-1 else 'cpu')


if args.dataset == 'mnist':
	train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', 
							train=True, transform=transforms.ToTensor(), download=True),
							batch_size=args.batch_size_test, shuffle=False, num_workers=4, drop_last=True)
	test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', 
							train=False, transform=transforms.ToTensor(), download=True), 
							batch_size=args.batch_size_test, shuffle=False, num_workers=4, drop_last=True)
	args.c, args.w, args.h = 1, 28, 28

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(name, fmt=':f'):
        name = name
        fmt = fmt
        reset()

    def reset(self):
        val = 0
        avg = 0
        sum = 0
        count = 0

    def update(val, n=1):
        val = val
        sum += val * n
        count += n
        avg = sum / count

    def __str__(self):
        fmtstr = '{name} {val' + fmt + '} ({avg' + fmt + '})'
        return fmtstr.format(**__dict__)

def score_robustness(S):
	S = torch.abs(S)
	lambda_max = S[:,0]
	S_norm = S/S.max(1)[0].unsqueeze(1)
	von_entr = (S_norm * torch.log(S_norm+1e-10)).sum(1)*(-1)
	return von_entr.detach().cpu(), lambda_max.detach().cpu()

def sample_eigen(vae, x_sample, output_path, bidx, iterations):
	vae.eval()
	x_recon = vae(x_sample)[0]
	eigen_x_step, eigen_x_recon_step, eigen_z_step, error_eigen_steps, U, S = attack(vae, x_sample, iterations)
	von_entr, lambda_max  = score_robustness(S)
	steps, eigen, batch, channels, width, height = eigen_x_recon_step.shape
	rows = int(batch**0.5)
	utils.save_image(x_sample, nrow=rows, fp=os.path.join(output_path, f"original_batch{bidx+1}.png"))
	utils.save_image(x_recon, nrow=rows, fp=os.path.join(output_path, f"reconstructed_batch{bidx+1}.png"))
	pfw.set_config(batch_size=x_sample.shape[0], device=x_sample.device)
	recon_fid = pfw.fid(x_recon.expand(-1, 3, -1, -1), real_images=x_sample.expand(-1, 3, -1, -1))
	print(f'Iters, Total step size, FID(clean, adv), FID(clean, cleanrecon), FID(clean, advrecon)')
	for step in range(steps):
		for direction in range(1):
			utils.save_image(eigen_x_recon_step[step,direction], nrow=rows,
						fp=os.path.join(output_path, f"recon_eigendirection{direction+1}_step{step}_batch{bidx+1}.png"))
			utils.save_image(eigen_x_step[step,direction], nrow=rows,
						fp=os.path.join(output_path, f"corrupted_eigendirection{direction+1}_step{step}_batch{bidx+1}.png"))
			adv_fid = pfw.fid(eigen_x_step[step,direction].expand(-1, 3, -1, -1), real_images=x_sample.expand(-1, 3, -1, -1))
			advrecon_fid = pfw.fid(eigen_x_recon_step[step,direction].expand(-1, 3, -1, -1), real_images=x_sample.expand(-1, 3, -1, -1))
			print(f' {args.iters} \t{np.logspace(-2, 1, 10)[step]:.4f}\t{adv_fid:.4f}  \t{recon_fid:.4f} \t{advrecon_fid:.4f}')
	return error_eigen_steps.detach().cpu(), von_entr, lambda_max

def attack(vae, x, iterations, nm_eigen=2, samples=8):
	vae.eval()
	batch, channels, width, height = x.shape		
	# reshape_pullback = x.view(batch, channels*width*height, -1)
			
	steps = np.logspace(-2, 1, 10)
	#steps = [steps[i] for i in range(len(steps)) if i%2==0]
	eigen_x_step, eigen_z_step, eigen_x_recon_step = [], [], []
	x_flat = x.view(-1, channels*width*height)
	error_eigen_steps = []
	for step in steps:
		# print(step)
		eigen_x, eigen_z, eigen_x_recon, error_eigen = [], [], [], []
		for eigen in range(nm_eigen):
			x_recon = x
			for iter in range(iterations):
				U, S = pull_back_eigen(vae, x_recon)
				U = U.to(x.device)
				S = S.to(x.device)
				x_epsilon = x_flat + step/iterations * torch.einsum('i, ij -> ij', S[:,eigen], U[:,:,eigen])
				x_epsilon = x_epsilon.view(-1, channels, width, height)
				z_epsilon, _ = vae.encode_forward(x_epsilon) # Use mean estimate
				x_recon = vae.decode_forward(z_epsilon)
			eigen_z.append(z_epsilon)
			eigen_x.append(x_epsilon)
			eigen_x_recon.append(x_recon)
			error_eigen.append(F.mse_loss(x, x_recon, reduction='mean').mean())
		error_eigen = torch.stack(error_eigen)
		error_eigen_steps.append(error_eigen)
		eigen_z = torch.stack(eigen_z)
		eigen_x = torch.stack(eigen_x)

		eigen_x_recon = torch.stack(eigen_x_recon)
		eigen_x_step.append(eigen_x)

		eigen_x_recon_step.append(eigen_x_recon)
		eigen_z_step.append(eigen_z)

	# steps x eigen x batch x channels x width x height
	eigen_x_recon_step = torch.stack(eigen_x_recon_step)
	eigen_x_step = torch.stack(eigen_x_step)
	eigen_z_step = torch.stack(eigen_z_step)
	error_eigen_steps = torch.stack(error_eigen_steps)
	return eigen_x_step, eigen_x_recon_step, eigen_z_step, error_eigen_steps, U, S

def visualise_eigenmax(dataloader, output_path, device):
	vae.eval()

	data_all, lambda_max, labels_all = [], [], []
	for x_sample, y_sample in dataloader:
		x_sample, y_sample = x_sample.to(device), y_sample.to(device)

		pullback = pull_back(x_sample)
		s1 = torch.symeig(pullback, eigenvectors=False)
		lambda_max.append(s1[:,0])
		data_all.append(x_sample)
		labels_all.append(y_sample)

	data_all = torch.stack(data_all)
	lambda_max = torch.stack(lambda_max)
	labels_all = torch.stack(labels_all)
	data_all = data_all.view(-1, 784).numpy()
	lambda_max = lambda_max.view(-1, 1)
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler
	pca = PCA(n_components=2)
	data_all = StandardScaler().fit_transform(data_all)
	data_pca = pca.fit_transform(data_all)

	fig, axs = plt.subplots(2, 1)
	axs[0,0].scatter(data_pca[:,0], data_pca[:,1], c=labels_all.flatten().numpy())
	axs[0,0].set_xlabel('PC 1')
	axs[0,0].set_ylabel('PC 2')
	axs[0,0].set_xlabel('PCA Data')
	axs[1,0].scatter(data_pca[:,0], data_pca[:,1], c=lambda_max.flatten().numpy())
	axs[1,0].set_xlabel('PC 1')
	axs[1,0].set_xlabel('PC 1')
	axs[1,0].set_title('lambda max')
	plt.savefig(f"{output_path}/pca_lamdamax.png")


def pull_back_eigen(vae, x, option='ltoi', stochasticG=True):
	b, c, w, h = x.shape
	x = x.requires_grad_(True)
	mu, logvar = vae.encode_forward(x)
	sig = torch.exp(0.5*logvar)

	if option == 'ltoi':
		if stochasticG:
			Jxmu, Jxsigma = [], []
			for i in range(mu.shape[1]):
				Jxmu.append(autograd.grad(mu[:,i], x, mu[:,i].data.new(mu[:,i].shape).fill_(1), create_graph=True)[0])
				Jxsigma.append(autograd.grad(sig[:,i], x, sig[:,i].data.new(sig[:,i].shape).fill_(1), create_graph=True)[0])

			Jxmu = torch.stack(Jxmu, -1).detach()
			Jxsigma = torch.stack(Jxsigma, -1).detach() # B x C x W x H x L
			Jxmu = Jxmu.view(b, c*w*h, -1)
			Jxsigma = Jxsigma.view(b, c*w*h, -1)
			Gxz = torch.einsum('bil,bjl->bij', Jxmu, Jxmu) + torch.einsum('bil,bjl->bij', Jxsigma, Jxsigma)
			S, U = np.linalg.eigh(Gxz.cpu().numpy())  # (B, L) (B, L, L) eigenvalues in ascending order
			S = np.flip(S, 1)
			U = np.flip(U, 1)
			U = torch.from_numpy(U.copy()).to(x.device)
			S = torch.from_numpy(S.copy()).to(x.device)
			# print(U.shape, S.shape) # (B, L, L) (B, L)
		else:
			Jxmu, Jxsigma = [], []
			for i in range(mu.shape[1]):
				Jxmu.append(autograd.grad(mu[:,i], x, mu[:,i].data.new(mu[:,i].shape).fill_(1), create_graph=True)[0])
			Jxmu = torch.stack(Jxmu, -1).detach()
			U, S, V = torch.svd(Jxmu.view(b, c*w*h, -1).cpu(), some=False, compute_uv=True)
			# print(U.shape, S.shape) # (B, C x W x H, C x W x H) (B, L)
	elif option == 'otoi':
		x_recon = vae.decode_forward(mu)
		x_recon = rearrange(x_recon, 'b c w h -> b (c w h)')
		if stochasticG:
			zJxmu, zJxsigma = [], []
			for i in range(mu.shape[1]):
				zJxmu.append(autograd.grad(x_recon[:,i], mu, x_recon[:,i].data.new(x_recon[:,i].shape).fill_(1), create_graph=True)[0])
				zJxsigma.append(autograd.grad(x_recon[:,i], sig, x_recon[:,i].data.new(x_recon[:,i].shape).fill_(1), create_graph=True)[0])

			zJxmu = torch.stack(zJxmu, -1)
			zJxsigma = torch.stack(zJxsigma, -1)
			zGxz = torch.einsum('bdn,bdm->bnm', zJxmu, zJxmu) + torch.einsum('bdn,bdm->bnm', zJxsigma, zJxsigma)
			zGxz = zGxz.detach()


			Jxmu, Jxsigma = [], []
			for i in range(mu.shape[1]):
				Jxmu.append(autograd.grad(mu[:,i], x, mu[:,i].data.new(mu[:,i].shape).fill_(1), create_graph=True)[0])
				Jxsigma.append(autograd.grad(sig[:,i], x, sig[:,i].data.new(sig[:,i].shape).fill_(1), create_graph=True)[0])

			Jxmu = torch.stack(Jxmu, -1).detach()
			Jxsigma = torch.stack(Jxsigma, -1).detach()

			xGxz = torch.einsum('bdn, bnm, bdm->bnm', zJxmu, zGxz, zJxmu) + torch.einsum('bdn, bnm, bdm->bnm', zJxsigma, zGxz, zJxsigma)
			xGxz = zGxz.detach()

			S, U = np.linalg.eigh(xGxz.detach().cpu().numpy())
			idx = S.argsort()[::-1]
			U, S = U[:,idx], S[:,idx]
		else:
			x_recon = vae.decode_forward(mu)
			x_recon = rearrange(x_recon, 'b c w h -> b (c w h)')
			#Jxmu, Jxsigma = [], []
			Jxrecon = []
			for i in range(x_recon.shape[1]):
				Jxrecon.append(autograd.grad(x_recon[:,i], x, x_recon[:,i].data.new(x_recon[:,i].shape).fill_(1), create_graph=True)[0])
				#Jxsigma.append(autograd.grad(sig[:,i], x, sig[:,i].data.new(sig[:,i].shape).fill_(1), create_graph=True)[0])

			Jxrecon = torch.stack(Jxrecon, -1).detach()
			
			U, S, V = torch.svd(Jxrecon.view(b, c*w*h, -1).cpu(), some=False, compute_uv=True)
	return U, S


kl_betas = [1e-3]
output_results = args.output_results  
output_model = args.output_model
for kl in kl_betas:
	# vae = VarAutoEncoder(dimensions=2,
    #         in_shape=(args.c, args.w, args.h),  # image spatial shape
    #         out_channels=1,
    #         latent_size=32,
    #         channels=(2, 4, 8),
    #         strides=(1, 2, 2),).to(device)
	vae = MLPVAE().to(device)
	output_results = os.path.join(args.output_results, args.dataset,  f"kl_{kl}", 'attack', str(args.iters))
	os.makedirs(output_results, exist_ok=True)
	output_model =  os.path.join(args.output_model,  args.dataset, f"kl_{kl}") 
	model_path = os.path.join(output_model, "50.pt") 
	model_sd = torch.load(model_path)['vae']
	vae.load_state_dict(model_sd)
	vae.eval()
	for i, (data, label) in enumerate(test_loader, 0):
		if i > 0: break
		data = data.to(device)
		error_eigen, entr, maxeigen = sample_eigen(vae, data, os.path.join(output_results), i, args.iters)



