from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from preresnet import PreActResNet34
#import models.imagenet as customized_models
from progress_dir.progress.bar import Bar as Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.nn.functional import normalize

"""default_model_names = sorted(name for name in models.__dict__
			if name.islower() and not name.startswith("__")
			and callable(models.__dict__[name]))

print (default_model_names)"""

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='/usr/local/Data/ILSVRC2012/', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=120, type=int, metavar='N',
help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
#parser.add_argument('--train-batch', default=448, type=int, metavar='N',
#                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90, 120],
		help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
		metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--ortho-decay', '--od', default=1e-6, type=float,
                    help = 'ortho weight decay')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=2, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
			help='use pre-trained model')

#Device options
parser.add_argument('--ngpu', default='4', type=int,
help='Number of GPUs used')

"""Function used for Orthogonal Regularization"""

def l2_reg_ortho(mdl):
	l2_reg = None
	for W in mdl.parameters():
		if W.ndimension() < 2:
			continue
		else:   
			cols = W[0].numel()
			rows = W.shape[0]
			w1 = W.view(-1,cols)
			wt = torch.transpose(w1,0,1)
			m  = torch.matmul(wt,w1)
			ident = Variable(torch.eye(cols,cols))
			ident = ident.cuda()

			w_tmp = (m - ident)
			height = w_tmp.size(0)
			u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
			v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
			u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
			sigma = torch.dot(u, torch.matmul(w_tmp, v))

			if l2_reg is None:
				l2_reg = (sigma)**2
			else:
				l2_reg = l2_reg + (sigma)**2
	return l2_reg


use_cuda = torch.cuda.is_available()
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

best_acc = 0

def main():
	global best_acc
	start_epoch = args.start_epoch
	
	#Data  Loader
	traindir = os.path.join(args.data, 'train')
	valdir = os.path.join(args.data, 'val')
	normaliz = transforms.Normalize(mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225])

	train_loader = torch.utils.data.DataLoader(
        	datasets.ImageFolder(traindir, transforms.Compose([
            	transforms.RandomResizedCrop(224),
            	transforms.RandomHorizontalFlip(),
            	transforms.ToTensor(),
            	normaliz,
		])),batch_size=args.train_batch, shuffle=True,
		num_workers=args.workers, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(
        	datasets.ImageFolder(valdir, transforms.Compose([
            	transforms.Resize(256),
            	transforms.CenterCrop(224),
            	transforms.ToTensor(),
            	normaliz,
        	])),batch_size=args.test_batch, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	#Create Model	
	model = PreActResNet34()

        #Get the number of model parameters
	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))
		
	model = torch.nn.DataParallel(model).cuda() 
	#cudnn.benchmark = True
	
	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
				nesterov = args.nesterov,weight_decay=args.weight_decay)
	
	title = 'ImageNet-' + 'PreActResNet34'
	if args.resume:
        	# Load checkpoint.
        	print('==> Resuming from checkpoint..')
        	assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        	args.checkpoint = os.path.dirname(args.resume)
        	checkpoint = torch.load(args.resume)
        	best_acc = checkpoint['best_acc']
        	start_epoch = checkpoint['epoch']
        	model.load_state_dict(checkpoint['state_dict'])
        	optimizer.load_state_dict(checkpoint['optimizer'])
        	logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
	else:
		logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
		logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc 1.', 'Valid Acc 1.', 'Train Acc 5.', 'Valid Acc 5.'])

	if args.evaluate:
		print('\nEvaluation only')
		test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
		print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
		return
	
	# Train and val
	for epoch in range(start_epoch, args.epochs):
        	adjust_learning_rate(optimizer, epoch)

        	print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        	#Adjust Orhto decay rate
        	odecay = adjust_ortho_decay_rate(epoch+1)
	
        	train_loss, train_acc, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, use_cuda,odecay)
        	test_loss, test_acc,test_acc5 = test(val_loader, model, criterion, epoch, use_cuda)
		
        	# append logger file
        	logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, train_acc5, test_acc5])

        	# save model
        	is_best = test_acc > best_acc
        	best_acc = max(test_acc, best_acc)
        	save_checkpoint({
                	'epoch': epoch + 1,
                	'state_dict': model.state_dict(),
                	'acc': test_acc,
                	'best_acc': best_acc,
                	'optimizer' : optimizer.state_dict(),
			}, is_best, checkpoint=args.checkpoint)

	logger.close()
	logger.plot()
	savefig(os.path.join(args.checkpoint, 'log.eps'))
	print('Best acc:')
	print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda,odecay):
	# switch to train mode
    	model.train()
    	batch_time = AverageMeter()
    	data_time = AverageMeter()
    	losses = AverageMeter()
    	top1 = AverageMeter()
    	top5 = AverageMeter()
    	end = time.time()

    	bar = Bar('Processing', max=len(train_loader))
    	for batch_idx, (inputs, targets) in enumerate(train_loader):
        	# measure data loading time
        	data_time.update(time.time() - end)

        	if use_cuda:
            		inputs, targets = inputs.cuda(), targets.cuda(async=True)
        	inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        	# compute output
        	outputs = model(inputs)
        	oloss = l2_reg_ortho(model)
        	oloss = odecay * oloss
        	loss = criterion(outputs, targets)
        	loss = loss + oloss 

        	# measure accuracy and record loss
        	prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        	losses.update(loss.item(), inputs.size(0))
        	top1.update(prec1.item(), inputs.size(0))
        	top5.update(prec5.item(), inputs.size(0))

        	# compute gradient and do SGD step
        	optimizer.zero_grad()
        	loss.backward()
        	optimizer.step()

        	# measure elapsed time
        	batch_time.update(time.time() - end)
        	end = time.time()

        	# plot progress
        	bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | t			op1: {top1: .4f} | top5: {top5: .4f}'.format(
                    	batch=batch_idx + 1,
                    	size=len(train_loader),
                    	data=data_time.val,
                    	bt=batch_time.val,
                    	total=bar.elapsed_td,
                    	eta=bar.eta_td,
                    	loss=losses.avg,
                    	top1=top1.avg,
                    	top5=top5.avg,
                    	)
        	bar.next()
    	bar.finish()
    	return (losses.avg, top1.avg, top5.avg)
	
def test(val_loader, model, criterion, epoch, use_cuda):
    	global best_acc

    	batch_time = AverageMeter()
    	data_time = AverageMeter()
    	losses = AverageMeter()
    	top1 = AverageMeter()
    	top5 = AverageMeter()

    	# switch to evaluate mode
    	model.eval()

    	end = time.time()
    	bar = Bar('Processing', max=len(val_loader))
    	for batch_idx, (inputs, targets) in enumerate(val_loader):
        	# measure data loading time
        	data_time.update(time.time() - end)

        	if use_cuda:
            		inputs, targets = inputs.cuda(), targets.cuda()
        	#inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        	with torch.no_grad():
        		inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        	# compute output
        	outputs = model(inputs)
        	loss = criterion(outputs, targets)

        	# measure accuracy and record loss
        	prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        	losses.update(loss.item(), inputs.size(0))
       		top1.update(prec1.item(), inputs.size(0))
        	top5.update(prec5.item(), inputs.size(0))

        	# measure elapsed time
        	batch_time.update(time.time() - end)
        	end = time.time()

        	# plot progress
        	bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    	batch=batch_idx + 1,
                    	size=len(val_loader),
                    	data=data_time.avg,
                    	bt=batch_time.avg,
                    	total=bar.elapsed_td,
                    	eta=bar.eta_td,
                    	loss=losses.avg,
                    	top1=top1.avg,
                    	top5=top5.avg,
                    	)
        	bar.next()
    	bar.finish()
    	return (losses.avg, top1.avg, top5.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    	filepath = os.path.join(checkpoint, filename)
    	torch.save(state, filepath)
    	if is_best:
        	shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    	global state
    	if epoch in args.schedule:
        	state['lr'] *= args.gamma
    	for param_group in optimizer.param_groups:
            	param_group['lr'] = state['lr']

def adjust_ortho_decay_rate(epoch):
    	o_d = args.ortho_decay

    	if epoch > 120:
        	o_d = 0.0
    	elif epoch > 70:
        	o_d = 1e-6 * o_d
    	elif epoch > 50:
        	o_d = 1e-4 * o_d
    	elif epoch > 30:
        	o_d = 1e-3 * o_d

    	return o_d


if __name__ == '__main__':
	main()
