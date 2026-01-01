# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from .BERT import BERTEmbedding, TransformerBlock, ScheduledOptim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import tqdm
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
from torch.utils.data import random_split

torch.manual_seed(1996)


vocab2index = {
	'[PAD]':0,
	'[SEP]':1,
	'[MASK]':2,
	'[CLS]':3,
	'[UNK]':4,
	'A':5,
	'C':6,
	'D':7,
	'E':8,
	'F':9,
	'G':10,
	'H':11,
	'I':12,
	'K':13,
	'L':14,
	'M':15,
	'N':16,
	'P':17,
	'Q':18,
	'R':19,
	'S':20,
	'T':21,
	'V':22,
	'W':23,
	'Y':24,
	}

class_list1 = pd.DataFrame(['TRA','TRB','Antigen peptide'])
class_list2 = pd.DataFrame(["Autoimmune", "Cancer", "Pathogens","Epitope"])
class_list3 = pd.DataFrame(["Autoimmune", "Cancer", "Pathogens","Epitope","Cross","Other"])

def shuffle_string(s):
	s_list = list(s)
	random.shuffle(s_list)
	return("".join(s_list))


class PRETrainData(torch.utils.data.Dataset):

	def __init__(self, sequence,label,speci_label,speci_prob, max_len = 48):
		
		self.max_len = max_len

		self.sentence = list(sequence)   
		self.label = list(label)
		self.speci_label = list(speci_label)
		self.speci_prob = np.array(speci_prob).astype(np.float32)
		
	def __len__(self):
		return len(self.sentence)

	def get_x(x,l):
		if x<int(l/2):
			x = x
		else:
			x = l-x


	def __getitem__(self, index):

		output_label = torch.zeros(self.max_len, dtype=torch.long)
		output_id = torch.zeros(self.max_len, dtype=torch.long)
		
		for i in range(self.max_len):
			if i == 0:
				output_id[i] = vocab2index['[CLS]']
				output_label[i] = 0
			elif i <= len(self.sentence[index]):
				prob = random.random()
				if prob < 0.15:
					#if prob < 0.15*get_x(i,len(self.sentence[index]))/len(self.sentence[index]):
					#if prob < 0.15*0.9:
					output_id[i] = vocab2index['[MASK]']
					# elif prob < 0.15*0.9:
					# 	output_id[i] = random.randrange(len(vocab2index))
					#else:
					#	output_id[i] = random.randrange(len(vocab2index))
						#output_id[i] = vocab2index[self.sentence[index][i-1]]
					#else:
					#	output_id[i] = random.randrange(len(vocab2index))
					output_label[i] = vocab2index[self.sentence[index][i-1]]
				else:
					output_id[i] = vocab2index[self.sentence[index][i-1]]
					output_label[i] = 0
			elif i == len(self.sentence[index]) + 1:
				output_id[i] = vocab2index['[SEP]']
				output_label[i] = 0
			else:
				output_id[i] = vocab2index['[PAD]']
				output_label[i] = 0
		return(output_id, output_label, torch.tensor(self.label[index]), torch.tensor(self.speci_label[index]), torch.tensor(self.speci_prob[index]),self.sentence[index])		

class GetData(torch.utils.data.Dataset):
	def __init__(self, sequence,label,speci_label,speci_prob, max_len = 48):
		self.max_len = max_len
		self.sentence = list(sequence)
		self.label = list(label)
		self.speci_label = list(speci_label) 
		self.speci_prob = np.array(speci_prob)
		
	def __len__(self):
		return len(self.sentence)

	def get_x(x,l):
		if x<int(l/2):
			x = x
		else:
			x = l-x

	def __getitem__(self, index):
		output_id = torch.zeros(self.max_len, dtype=torch.long)		
		for i in range(self.max_len):
			if i == 0:
				output_id[i] = vocab2index['[CLS]']
			elif i <= len(self.sentence[index]):
				output_id[i] = vocab2index[self.sentence[index][i-1]]
			elif i == len(self.sentence[index]) + 1:
				output_id[i] = vocab2index['[SEP]']
			else:
				output_id[i] = vocab2index['[PAD]']
		return(output_id,torch.tensor(self.label[index]),torch.tensor(self.speci_label[index]), torch.tensor(self.speci_prob[index]),self.sentence[index])

class GetData_seq(torch.utils.data.Dataset):
	def __init__(self, sequence, max_len = 48):
		self.max_len = max_len
		self.sentence = list(sequence)   
		
	def __len__(self):
		return len(self.sentence)

	def get_x(x,l):
		if x<int(l/2):
			x = x
		else:
			x = l-x

	def __getitem__(self, index):
		output_id = torch.zeros(self.max_len, dtype=torch.long)		
		for i in range(self.max_len):
			if i == 0:
				output_id[i] = vocab2index['[CLS]']
			elif i <= len(self.sentence[index]):
				if self.sentence[index][i-1] in vocab2index.keys():
					output_id[i] = vocab2index[self.sentence[index][i-1]]
				else:
					output_id[i] = vocab2index['[MASK]']
			elif i == len(self.sentence[index]) + 1:
				output_id[i] = vocab2index['[SEP]']
			else:
				output_id[i] = vocab2index['[PAD]']
		return(output_id,self.sentence[index])


class BERT_encoder(nn.Module):
	"""
	BERT model : Bidirectional Encoder Representations from Transformers.
	"""

	def __init__(self, vocab_size, hidden=768, n_layers=8, attn_heads=12, dropout=0.1):
		"""
		:param vocab_size: vocab_size of total words
		:param hidden: BERT model hidden size
		:param n_layers: numbers of Transformer blocks(layers)
		:param attn_heads: number of attention heads
		:param dropout: dropout rate
		"""

		super().__init__()
		self.hidden = hidden
		self.n_layers = n_layers
		self.attn_heads = attn_heads

		# paper noted they used 4*hidden_size for ff_network_hidden_size
		self.feed_forward_hidden = hidden * 4

		# embedding for BERT, sum of positional, segment, token embeddings
		self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

		# multi-layers transformer blocks, deep network
		self.transformer_blocks = nn.ModuleList(
			[TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
		

	def forward(self, x):
		# attention masking for padded token
		# torch.ByteTensor([batch_size, 1, seq_len, seq_len)
		mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
		self.mask = mask

		# embedding the indexed sequence to sequence of vectors
		x = self.embedding(x)

		# running over multiple transformer blocks
		for transformer in self.transformer_blocks:
			x = transformer.forward(x, mask)
			
		return x	
	
class FocalLoss(nn.Module):
	r"""
		This criterion is a implemenation of Focal Loss, which is proposed in 
		Focal Loss for Dense Object Detection.

			Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

		The losses are averaged across observations for each minibatch.

		Args:
			alpha(1D Tensor, Variable) : the scalar factor for this criterion
			gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
								   putting more focus on hard, misclassiﬁed examples
			size_average(bool): By default, the losses are averaged over observations for each minibatch.
								However, if the field size_average is set to False, the losses are
								instead summed for each minibatch.


	"""
	def __init__(self, class_num, alpha=None, gamma=2, size_average=True,device="cuda"):
		super(FocalLoss, self).__init__()
		if alpha is None:
			self.alpha = Variable(torch.ones(class_num, 1)).to(device)
		else:
			if isinstance(alpha, Variable):
				self.alpha = alpha.to(device)
			else:
				self.alpha = Variable(alpha).to(device)
		self.gamma = gamma
		self.class_num = class_num
		self.size_average = size_average
		self.device = device

	def forward(self, inputs, targets):
		N = inputs.size(0)
		C = inputs.size(1)
		P = F.softmax(inputs,dim=1)

		class_mask = inputs.data.new(N, C).fill_(0)
		class_mask = Variable(class_mask)
		ids = targets.view(-1, 1)
		class_mask.scatter_(1, ids.data, 1.) #(N,C) 1,0
		#print(class_mask)


		#if inputs.is_cuda and not self.alpha.is_cuda:
		#self.alpha = self.alpha#.to(device)
		#print(ids.data.shape)
		alpha = self.alpha[ids.data.view(-1)].to(self.device)

		probs = (P*class_mask).sum(1).view(-1,1).to(self.device)

		log_p = probs.log().to(self.device)

		batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

		if self.size_average:
			loss = batch_loss.mean()
		else:
			loss = batch_loss.sum()
		return loss

class ProbabilisticFocalLoss(nn.Module):
	r"""
		This criterion is a implemenation of Focal Loss, which is proposed in 
		Focal Loss for Dense Object Detection.

			Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

		The losses are averaged across observations for each minibatch.

		Args:
			alpha(1D Tensor, Variable) : the scalar factor for this criterion
			gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
								   putting more focus on hard, misclassiﬁed examples
			size_average(bool): By default, the losses are averaged over observations for each minibatch.
								However, if the field size_average is set to False, the losses are
								instead summed for each minibatch.


	"""
	def __init__(self, class_num, alpha=None, gamma=2, size_average=True,device="cuda"):
		super(FocalLoss, self).__init__()
		if alpha is None:
			self.alpha = Variable(torch.ones(class_num, 1)).to(device)
		else:
			if isinstance(alpha, Variable):
				self.alpha = alpha.to(device)
			else:
				self.alpha = Variable(alpha).to(device)
		self.gamma = gamma
		self.class_num = class_num
		self.size_average = size_average
		self.device = device

	def forward(self, inputs, targets):
		N = inputs.size(0)
		C = inputs.size(1)
		P = F.softmax(inputs,dim=1)
		alpha = self.alpha[targets.argmax(dim=1)]#.to(self.device)

		ce_loss = -targets * torch.log(P)  # (N, C)
		modulating_factor = torch.pow((1 - P), self.gamma)  # (N, C)
		fl_loss = alpha * modulating_factor * ce_loss  # (N, C)
		batch_loss = fl_loss.sum(dim=-1)  # (N,)

		class_mask = Variable(targets).to(self.device)
		probs = (P*class_mask).sum(1).view(-1,1)#.to(self.device)
		log_p = probs.log()#.to(self.device)

		batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

		if self.size_average:
			loss = batch_loss.mean()
		else:
			loss = batch_loss.sum()
		return loss


class WeightedKLDivLoss(nn.Module):
	def __init__(self, class_num, gamma=2, size_average=True,device="cuda"):
		super().__init__()
		#self.class_weights = class_weights  # 形状: [num_classes]
		self.beta = Variable(torch.ones(class_num, 1)).to(device)

		self.gamma = gamma

		self.class_num = class_num
		self.size_average = size_average
		self.device = device

	def forward(self, inputs, targets):
		# input: 模型输出的log概率（需log_softmax）
		# target: 平滑后的概率分布
		# kl_loss = F.kl_div(inputs, targets, reduction='none')
		# weighted_loss = kl_loss * self.beta[targets.argmax(dim=1)]

		"""
		Args:
			input (torch.Tensor): 预测概率分布 (batch_size, num_classes)
			target (torch.Tensor): 真实分布 (batch_size, num_classes)，可以是one-hot或soft label
		"""
		# 计算 KL 散度的基本项：P * log(P / Q)
		kl_div = targets * (torch.log(targets + 1e-8) - torch.log_softmax(inputs, dim=1))

		# 计算 Focal 权重：|log(P / Q)|^gamma
		log_ratio = torch.log(targets + 1e-8) - torch.log_softmax(inputs, dim=1)

		beta = self.beta[targets.argmax(dim=1)].to(self.device)

		focal_weight = beta*torch.abs(log_ratio).pow(self.gamma)

		# 加权 KL 散度
		focal_kl = focal_weight * kl_div

		if self.size_average:
			loss = focal_kl.mean()
		else:
			loss = focal_kl.sum()

		return loss

class TCREPbert(nn.Module):
	
	def __init__(self, vocab_size=25, max_len = 48, hidden=512, n_layers=8, attn_heads=8, dropout=0.3,output_dim1=3,output_dim2=4, device='cpu'):

		super().__init__()
		
		self.setup_seed(1996)
		self.max_len = max_len
		self.hidden = hidden
		self.device = device
		
		self.dropout = nn.Dropout(dropout)

		self.BERT_model = BERT_encoder(vocab_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads,dropout=dropout)
		
		self.hidden2weight = nn.Linear(hidden, 1)
		self.criterion1 = nn.NLLLoss(ignore_index=0) #ignore_index设定labels中需要忽略的值，一般为填充值，即该类别的误差不计入loss, 默认为-100
		self.criterion2 = nn.NLLLoss()
		self.hidden2vocab = nn.Linear(hidden, vocab_size)
		self.hidden2type1 = nn.Linear(hidden, output_dim1)
		self.hidden2type2 = nn.Linear(hidden, output_dim2)
		self.logsoftmax = nn.LogSoftmax(dim=-1)
		self.softmax = nn.Softmax(dim=-1)
		self.Focal_Loss1 = FocalLoss(class_num=output_dim1,device=device)
		# self.Focal_Loss2 = FocalLoss(class_num=output_dim2,device=device)
		#self.KL_Loss = nn.KLDivLoss(reduction='batchmean')
		# self.KL_Loss = WeightedKLDivLoss(class_num=output_dim2,device=device)
	
	def setup_seed(self, seed):
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)
		torch.backends.cudnn.deterministic = True 

	def forward_label(self, x):
		preds1 = self.hidden2type1(self.dropout(x[:, 0]))
		score1 = self.softmax(preds1)
		return score1,preds1

	def Masked_Loss(self, x, label):
		#x:[batch,max_len,hidden]		
		x = self.logsoftmax(self.hidden2vocab(x))
		#x:[batch,max_len,vocab_size]
		#label:[batch,max_len]
		#x.transpose(1, 2)等同于x.permute(0,2,1).contiguous()
		loss = self.criterion1(x.permute(0,2,1).contiguous(), label)
		
		return loss
	
	def class_Loss(self, preds1, stype1):
		#x:[batch,max_len,hidden]
		#x[:,0]:[batch,hidden]
		#stype:[batch]

		# preds1 = self.hidden2type1(x[:, 0])
		# score1 = self.softmax(preds1)
		# preds2 = self.hidden2type2(x[:, 0])
		# score2 = self.softmax(preds2)
		# loss = self.Focal_Loss1(preds1,stype1) + 0.2*self.Focal_Loss2(preds2,stype2)
		loss = self.label_Loss(preds1,stype1) 
		
		return loss

	def label_Loss(self, preds1, stype1):
		#x:[batch,max_len,hidden]
		#x[:,0]:[batch,hidden]
		#stype:[batch]

		# preds1 = self.hidden2type1(self.dropout(x[:, 0]))
		# #preds1 = self.dropout(preds1)
		
		# score1 = self.softmax(preds1)

		#loss = self.criterion2(self.logsoftmax(self.hidden2type(x[:, 0])), stype)
		loss = self.Focal_Loss1(preds1,stype1)
		
		return loss
	
	def get_loss(self,x,label,stype1, preds1):
		mask_loss = self.Masked_Loss(x, label)
		class_loss = self.class_Loss(preds1, stype1)			
		loss = class_loss + mask_loss

		return loss

	def evaluate(self,test_loader,output_dir,prefix,device):
		prob_all1 = []
		prob_all2 = []
		prob_all3 = []
		prob_all0 = []

		label_all1 = []
		label_all2 = []
		label_all3 = []
		label_all0 = []

		self.eval()
		with torch.no_grad():
			for i, data in enumerate(test_loader):				
				input_sentence = data[0].to(device)
				input_stype1 = data[1].to(device)				
				input_stype2 = data[2].to(device)
				input_stype2_prob = data[3].to(device)
				
				output_hidden = self.BERT_model(input_sentence)
				mask_loss = self.Masked_Loss(output_hidden, input_sentence)
				
				# class_score1,class_score2,loss = model.get_loss(output_hidden,input_sentence,input_stype1,input_stype2_prob)
				#class_score1,class_score2,class_loss = model.class_Loss(output_hidden,input_stype1,input_stype2_prob)

				class_score1,preds1 = self.forward_label(output_hidden)
				class_loss = self.class_Loss(preds1,input_stype1)
				loss = class_loss + mask_loss
				
				predict_score1,predict1 = class_score1.max(dim=-1)
				correct1 = predict1.eq(input_stype1).sum().item()
				accuracy1 = correct1/input_stype1.nelement()
				predict_label1 = class_list1.iloc[predict1.cpu().numpy(),0]			

				input_label_class = (input_stype1!=2).long()
				pred_out_class = (predict1!=2).long()
				pred_12_score = 1-class_score1[:,2]
				
				input_label1 = (input_stype1==0).long()
				input_label2 = (input_stype1==1).long()
				input_label3 = (input_stype1==2).long()

				prob_all0.extend(pred_12_score.cpu().numpy())
				prob_all1.extend(class_score1[:,0].cpu().numpy())
				prob_all2.extend(class_score1[:,1].cpu().numpy())
				prob_all3.extend(class_score1[:,2].cpu().numpy())
				
				label_all0.extend(input_label_class.cpu().numpy())
				label_all1.extend(input_label1.cpu().numpy())
				label_all2.extend(input_label2.cpu().numpy())
				label_all3.extend(input_label3.cpu().numpy())
		

		#ROC curve for classify label
		class_auc0 = roc_auc_score(label_all0,prob_all0)
		class_auc1 = roc_auc_score(label_all1,prob_all1)
		class_auc2 = roc_auc_score(label_all2,prob_all2)
		class_auc3 = roc_auc_score(label_all3,prob_all3)

	def pretrain(self,
				 train_loader,
				 output_dir,
				 prefix,
				 epochs = 10, 
				 lr = 1e-5, 
				 betas = (0.9, 0.98), 
				 weight_decay = 0.01,
				 warmup_steps = 10000,
				 is_train = True):
		
		optim = Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
		optim_schedule = ScheduledOptim(optim, self.hidden, n_warmup_steps=warmup_steps)

		train_evaluation_df = dict()
		train_evaluation_df['epoch'] = []
		train_evaluation_df['loss'] = []
		train_evaluation_df['mask_loss'] = []
		train_evaluation_df['class_loss'] = []
		train_evaluation_df['accuracy1'] = []

		val_evaluation_df = dict()
		val_evaluation_df['epoch'] = []
		val_evaluation_df['batch'] = []
		val_evaluation_df['loss'] = []
		val_evaluation_df['mask_loss'] = []
		val_evaluation_df['class_loss'] = []
		val_evaluation_df['accuracy1'] = []
		
		for epoch in range(epochs):
			str_code = "train"
			
			data_iter = tqdm.tqdm(enumerate(train_loader),
								  desc="EP_%s:%d" % (str_code, epoch),
								  total=len(train_loader),
								  bar_format="{l_bar}{r_bar}")
			
			avg_loss = 0.0
			class_avg_loss = 0.0
			mask_avg_loss = 0.0
			avg_acc1 = 0.0
			self.train()
			for i, data in data_iter:
				
				input_sentence = data[0].to(self.device)
				input_label = data[1].to(self.device)
				input_stype1 = data[2].to(self.device)		
				input_stype2 = data[3].to(self.device)
				input_stype2_prob = data[4].to(self.device)

				print(input_sentence.size)
				print(input_label.size)
				
				output_hidden = self.BERT_model(input_sentence)
				#mask_loss = self.Masked_Loss(output_hidden, input_label)
				#class_score, class_loss = self.class_Loss(output_hidden, input_stype)
				
				#loss = 0.1*class_loss + mask_loss


				#class_score1,class_score2,loss = self.get_loss(output_hidden,input_label,input_stype1,input_stype2)
				mask_loss = self.Masked_Loss(output_hidden,input_label)

				#class_score1,class_score2,class_loss = self.class_Loss(output_hidden,input_stype1,input_stype2_prob)
				class_score1,preds1 = self.forward_label(output_hidden)
				class_loss = self.class_Loss(preds1,input_stype1)

				loss = mask_loss + class_loss
				
				optim_schedule.zero_grad()
				# if is_train:					
				# 	mask_loss.backward()
				# 	is_train = False
				# else:
				# 	class_loss.backward()
				# 	is_train = True
				mask_loss.backward(retain_graph=True)
				class_loss.backward()
				#loss.backward()
				
				torch.nn.utils.clip_grad_norm_(self.parameters(),1.0)
				optim_schedule.step_and_update_lr()
				
				avg_loss += loss.item()
				class_avg_loss += class_loss.item()
				mask_avg_loss += mask_loss.item()					

				correct1 = class_score1.argmax(dim=-1).eq(input_stype1).sum().item()
				accuracy1 = correct1/input_stype1.nelement()				

				avg_acc1 += accuracy1
			
			print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
			print("EP%d_%s, class_avg_loss=" % (epoch, str_code), class_avg_loss / len(data_iter))
			print("EP%d_%s, mask_avg_loss=" % (epoch, str_code), mask_avg_loss / len(data_iter))	
			print("EP%d_%s, avg_acc1=" % (epoch, str_code), avg_acc1 / len(data_iter))
			print("saving model EP%s:"%str(epoch))
			# output_path = os.path.join(output_dir, prefix+"_AntiSPencoder.ep%d" % epoch)
			# self.save_model(output_path)

			checkpoint = {
			    "epoch": epoch,
			    "model_state_dict": self.state_dict(),
			    # "optimizer_state_dict": optim.state_dict(),
			    "loss": avg_loss / len(data_iter),
			    "mask_loss": mask_avg_loss / len(data_iter),
			    "class_loss": class_avg_loss / len(data_iter),
			    "accuracy1": avg_acc1 / len(data_iter),
			}
			checkpoint_path = os.path.join(output_dir, prefix+"_AntiSPencoder_checkpoint.ep%d" % epoch)
			torch.save(checkpoint, checkpoint_path)

			train_evaluation_df['epoch'].append(epoch)
			train_evaluation_df['loss'].append(avg_loss / len(data_iter))
			train_evaluation_df['mask_loss'].append(mask_avg_loss / len(data_iter))
			train_evaluation_df['class_loss'].append(class_avg_loss / len(data_iter))
			train_evaluation_df['accuracy1'].append(avg_acc1 / len(data_iter))

			avg_loss_val = 0.0
			mask_avg_loss_val = 0.0
			class_avg_loss_val = 0.0
			avg_acc_val1 = 0.0
			prob_all1 = []
			prob_all2 = []
			prob_all3 = []
			prob_all0 = []

			label_all1 = []
			label_all2 = []
			label_all3 = []
			label_all0 = []


			str_code = "val"
			
			val_data_iter = tqdm.tqdm(enumerate(val_loader),
								  desc="EP_%s:%d" % (str_code, epoch),
								  total=len(val_loader),
								  bar_format="{l_bar}{r_bar}")

			self.eval()		
			with torch.no_grad():
				for i, data in val_data_iter:				
					input_sentence = data[0].to(self.device)
					input_stype1 = data[1].to(self.device)				
					input_stype2 = data[2].to(self.device)
					input_stype2_prob = data[3].to(self.device)
					print(input_sentence.size)

					output_hidden = self.BERT_model(input_sentence)
					#mask_loss = self.Masked_Loss(output_hidden, input_label)
					#class_score, class_loss = self.class_Loss(output_hidden, input_stype)
					
					#class_score1,class_score2,loss = self.get_loss(output_hidden,input_label,input_stype1,input_stype2)		
					mask_loss = self.Masked_Loss(output_hidden,input_sentence)
					#class_score1,class_score2,class_loss = self.class_Loss(output_hidden,input_stype1,input_stype2_prob)
					class_score1,preds1 = self.forward_label(output_hidden)
					class_loss = self.class_Loss(preds1,input_stype1)

					loss = mask_loss + class_loss

					avg_loss_val += loss.item()
					mask_avg_loss_val += mask_loss.item()
					class_avg_loss_val += class_loss.item()

					correct1 = class_score1.argmax(dim=-1).eq(input_stype1).sum().item()
					accuracy1 = correct1/input_stype1.nelement()

					
					avg_acc_val1 += accuracy1

					val_evaluation_df['epoch'].append(epoch)
					val_evaluation_df['batch'].append(i)
					val_evaluation_df['loss'].append(loss.item())
					val_evaluation_df['mask_loss'].append(mask_loss.item())
					val_evaluation_df['class_loss'].append(class_loss.item())
					val_evaluation_df['accuracy1'].append(accuracy1)


					input_label_class = (input_stype1!=2).long()
					pred_12_score = 1-class_score1[:,2]

					input_label1 = (input_stype1==0).long()
					input_label2 = (input_stype1==1).long()
					input_label3 = (input_stype1==2).long()

					prob_all0.extend(pred_12_score.cpu().numpy())
					prob_all1.extend(class_score1[:,0].cpu().numpy())
					prob_all2.extend(class_score1[:,1].cpu().numpy())
					prob_all3.extend(class_score1[:,2].cpu().numpy())
					
					label_all0.extend(input_label_class.cpu().numpy())
					label_all1.extend(input_label1.cpu().numpy())
					label_all2.extend(input_label2.cpu().numpy())
					label_all3.extend(input_label3.cpu().numpy())

					

			#ROC curve for classify label
			class_auc0 = roc_auc_score(label_all0,prob_all0)
			class_auc1 = roc_auc_score(label_all1,prob_all1)
			class_auc2 = roc_auc_score(label_all2,prob_all2)
			class_auc3 = roc_auc_score(label_all3,prob_all3)

			print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss_val / len(val_data_iter))
			print("EP%d_%s, class_avg_loss=" % (epoch, str_code), class_avg_loss_val / len(val_data_iter))
			print("EP%d_%s, mask_avg_loss=" % (epoch, str_code), mask_avg_loss_val / len(val_data_iter))			
			print("EP%d_%s, avg_acc1=" % (epoch, str_code), avg_acc_val1 / len(val_data_iter))
			print("EP%d_%s, class_auc0=" % (epoch, str_code), class_auc0)
			print("EP%d_%s, class_auc1=" % (epoch, str_code), class_auc1)
			print("EP%d_%s, class_auc2=" % (epoch, str_code), class_auc2)
			print("EP%d_%s, class_auc3=" % (epoch, str_code), class_auc3)	


		train_output_path = os.path.join(output_dir, prefix+"_AntiSPencoder_train_loss_accuracy_epoch%s.pt"%str(epochs))
		torch.save(train_evaluation_df, train_output_path)
		val_output_path = os.path.join(output_dir, prefix+"_AntiSPencoder_val_loss_accuracy_epoch%s.pt"%str(epochs))
		val_evaluation_df = pd.DataFrame(val_evaluation_df)
		torch.save(val_evaluation_df, val_output_path)
		return train_evaluation_df,val_evaluation_df
		
	def save_model(self, output_path):
		
		torch.save(self.cpu(), output_path)
		self.to(self.device)
		print("Model Saved on:", output_path)


	def predict(self,sequence,device='cpu',batch_size=32,num_workers=4):
		dataset = GetData_seq(sequence)
		data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
		encoder_info = pd.DataFrame()

		embeddings_info = pd.DataFrame()
		if self.training is True:
			print("Please set model.eval() while predicting!!")
			return "0"

		data_iter = tqdm.tqdm(enumerate(data_loader),
								  desc="Query",
								  total=len(data_loader),
								  bar_format="{l_bar}{r_bar}")
		self.eval()
		with torch.no_grad():
			for i, data in data_iter:				
				input_id = data[0].to(device)
				input_sentence = data[1]
				
				output_hidden = self.BERT_model(input_id)

				class_score1,preds1 = self.forward_label(output_hidden)
				
				predict_score1,predict1 = class_score1.max(dim=-1)
				predict_label1 = class_list1.iloc[predict1.cpu().numpy(),0]


				#x1 = self.hidden2vocab(output_hidden)
				#x1:[batch,max_len,vocab_size]
				#label:[batch,max_len]
				#x1.transpose(1, 2)等同于x1.permute(0,2,1).contiguous()
				#emb_loss = self.criterion1(x1.permute(0,2,1).contiguous(), input_id)
				
				c_df = pd.DataFrame({"Sequence":list(input_sentence),
					#"label":list(input_stype.cpu().numpy()),
					"predict_id1":predict1.cpu().numpy(),
					"predict_label1":list(predict_label1),
					"predict_score1":list(predict_score1.cpu().numpy())
					# "label0_score":list(class_score1[:,0].cpu().numpy()),
					# "label1_score":list(class_score1[:,1].cpu().numpy()),
					# "label2_score":list(class_score1[:,2].cpu().numpy()),
					# "speci0_score":list(class_score2[:,0].cpu().numpy()),
					# "speci1_score":list(class_score2[:,1].cpu().numpy()),
					# "speci2_score":list(class_score2[:,2].cpu().numpy()),
				})

				label_score1 = pd.DataFrame(class_score1.cpu().numpy())
				label_score1.columns = list(class_list1[0])
				c_df = pd.concat([c_df, label_score1], axis=1)  # 横向合并

				encoder_info = pd.concat([encoder_info,c_df],axis=0)
				
				c_embed = pd.DataFrame(output_hidden[:, 0].cpu().numpy())
				#c_embed.index = input_sentence
				c_embed.insert(0, 'sequence', list(input_sentence))
				embeddings_info = pd.concat([embeddings_info,c_embed],axis=0)

		#encoder_info.to_csv(save_file,sep="\t")
		return encoder_info,embeddings_info

def evaluate(model,test_loader,output_dir,prefix,device):
	evaluation_df = dict()
	evaluation_df['loss'] = []
	evaluation_df['accuracy'] = []
	evaluation_df['precision'] = []
	evaluation_df['recall'] = []
	test_result_info = pd.DataFrame()

	prob_all1 = []
	prob_all2 = []
	prob_all3 = []
	prob_all0 = []

	label_all1 = []
	label_all2 = []
	label_all3 = []
	label_all0 = []

	embeddings_info = pd.DataFrame()
	model.eval()
	with torch.no_grad():
		for i, data in enumerate(test_loader):				
			input_sentence = data[0].to(device)
			input_stype1 = data[1].to(device)				
			input_stype2 = data[2].to(device)
			input_stype2_prob = data[3].to(device)
			
			output_hidden = model.BERT_model(input_sentence)
			mask_loss = model.Masked_Loss(output_hidden, input_sentence)
			
			# class_score1,class_score2,loss = model.get_loss(output_hidden,input_sentence,input_stype1,input_stype2_prob)
			#class_score1,class_score2,class_loss = model.class_Loss(output_hidden,input_stype1,input_stype2_prob)

			class_score1,preds1 = model.forward_label(output_hidden)
			class_loss = model.class_Loss(preds1,input_stype1)
			loss = class_loss + mask_loss	
			
			predict_score1,predict1 = class_score1.max(dim=-1)
			correct1 = predict1.eq(input_stype1).sum().item()
			accuracy1 = correct1/input_stype1.nelement()
			predict_label1 = class_list1.iloc[predict1.cpu().numpy(),0]

			input_label_class = (input_stype1!=2).long()
			pred_out_class = (predict1!=2).long()
			pred_12_score = 1-class_score1[:,2]
			# print(input_stype)
			# print(input_label1)
			#auc = roc_auc_score(input_label1.cpu().numpy(),pred_out1.cpu().numpy())
			precision = precision_score(input_label_class.cpu().numpy(),pred_out_class.cpu().numpy())
			recall = recall_score(input_label_class.cpu().numpy(),pred_out_class.cpu().numpy())		

			evaluation_df['loss'].append(np.mean(loss.cpu().numpy()))
			evaluation_df['accuracy'].append(accuracy1)
			#evaluation_df['AUC'].append(auc)
			evaluation_df['precision'].append(precision)
			evaluation_df['recall'].append(recall)

			input_sequence = data[4]
			c_df = pd.DataFrame({"Sequence":list(input_sequence),
				"id":list(input_stype1.cpu().numpy()),
				"label":list(class_list1.iloc[input_stype1.cpu().numpy(),0]),
				"predict_score":list(predict_score1.cpu().numpy()),
				"predict_id1":list(predict1.cpu().numpy()),
				"predict_label1":list(predict_label1),
				"id2":list(input_stype2.cpu().numpy()),
				"label2":list(class_list3.iloc[input_stype2.cpu().numpy(),0])
				})
			label_score1 = pd.DataFrame(class_score1.cpu().numpy())
			label_score1.columns = list(class_list1[0])

			c_df = pd.concat([c_df, label_score1], axis=1)  # 横向合并
			test_result_info = pd.concat([test_result_info,c_df],axis=0)

			input_label1 = (input_stype1==0).long()
			input_label2 = (input_stype1==1).long()
			input_label3 = (input_stype1==2).long()

			prob_all0.extend(pred_12_score.cpu().numpy())
			prob_all1.extend(class_score1[:,0].cpu().numpy())
			prob_all2.extend(class_score1[:,1].cpu().numpy())
			prob_all3.extend(class_score1[:,2].cpu().numpy())
			
			label_all0.extend(input_label_class.cpu().numpy())
			label_all1.extend(input_label1.cpu().numpy())
			label_all2.extend(input_label2.cpu().numpy())
			label_all3.extend(input_label3.cpu().numpy())

			c_embed = pd.DataFrame(output_hidden[:, 0].cpu().numpy())
			c_embed.insert(0, 'sequence', list(input_sequence))
			embeddings_info = pd.concat([embeddings_info,c_embed],axis=0)
		
	evaluation_df = pd.DataFrame(evaluation_df)
	output_path = os.path.join(output_dir, prefix+"_AntiSPencoder_test_accuracy_info.txt")
	evaluation_df.to_csv(output_path,sep="\t")
	#torch.save(evaluation_df, output_path)
	#print(evaluation_df['recall'])
	#print(evaluation_df['precision'])

	predict_info_file = os.path.join(output_dir, prefix+"_AntiSPencoder_test_info.txt")
	test_result_info.to_csv(predict_info_file,sep="\t")

	embed_save_file = os.path.join(output_dir, prefix+"_AntiSPencoder_test_embeddings.txt")
	embeddings_info.to_csv(embed_save_file,sep="\t")

	plt.figure(None,(5,5))
	plt.title('testing loss')
	plt.boxplot(evaluation_df['loss'],showmeans=True,meanline=True,patch_artist=True,boxprops={'facecolor':'coral'})
	plt.xlabel("Embedding")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(output_dir,prefix+'_AntiSPencoder_test_loss_batch_boxplot.pdf'))


	evaluation_df_long = evaluation_df[['accuracy','precision','recall']].melt(var_name='Class', value_name='Score')
	custom_palette = {"accuracy":"#FF9999", "precision":"#66B2FF", "recall":"#99FF99"}
	plt.figure(None,(5,5))
	sns.boxplot(x='Class', y='Score', data=evaluation_df_long, palette=custom_palette, showfliers=False,order=list(custom_palette.keys()))
	# 添加标题和标签
	plt.title('Performance of encoder', fontsize=14)
	plt.xlabel('Metrics', fontsize=12)
	plt.ylabel('Score', fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.savefig(os.path.join(output_dir,prefix+'_AntiSPencoder_test_performance_batch_boxplot.pdf'))


	plt.figure(None,(5,5))
	plt.title('testing accuracy1')
	plt.boxplot(evaluation_df['accuracy'],showmeans=True,meanline=True,patch_artist=True,boxprops={'facecolor':'orange'})
	plt.xlabel("Embedding")
	plt.ylabel("Accuracy")
	plt.savefig(os.path.join(output_dir,prefix+'_AntiSPencoder_test_accuracy1_batch_boxplot.pdf'))

	# plt.figure(None,(5,5))
	# plt.title('testing loss')
	# plt.boxplot(evaluation_df['AUC'],showmeans=True,meanline=True,patch_artist=True,boxprops={'facecolor':'orange'})
	# plt.xlabel("Embedding")
	# plt.ylabel("AUC")
	# plt.savefig(os.path.join(output_dir,prefix+'_AntiSPencoder_test_auc_batch_boxplot.pdf'))

	plt.figure(None,(5,5))
	plt.title('testing recall')
	plt.boxplot(evaluation_df['precision'],showmeans=True,meanline=True,patch_artist=True,boxprops={'facecolor':'orange'})
	plt.xlabel("Embedding")
	plt.ylabel("AUC")
	plt.savefig(os.path.join(output_dir,prefix+'_AntiSPencoder_test_precision_batch_boxplot.pdf'))

	plt.figure(None,(5,5))
	plt.title('testing recall')
	plt.boxplot(evaluation_df['recall'],showmeans=True,meanline=True,patch_artist=True,boxprops={'facecolor':'orange'})
	plt.xlabel("Embedding")
	plt.ylabel("Recall")
	plt.savefig(os.path.join(output_dir,prefix+'_AntiSPencoder_test_recall_batch_boxplot.pdf'))

	#ROC curve for classify label
	auc0 = roc_auc_score(label_all0,prob_all0)
	fpr0,tpr0,thresholds0 = roc_curve(label_all0,prob_all0)
	auc1 = roc_auc_score(label_all1,prob_all1)
	fpr1,tpr1,thresholds1 = roc_curve(label_all1,prob_all1)
	auc2 = roc_auc_score(label_all2,prob_all2)
	fpr2,tpr2,thresholds2 = roc_curve(label_all2,prob_all2)
	auc3 = roc_auc_score(label_all3,prob_all3)
	fpr3,tpr3,thresholds3 = roc_curve(label_all3,prob_all3)

	plt.figure()
	plt.plot(fpr0, tpr0, color='#F04339', lw=2, label='TCR = %0.4f' % auc0)
	plt.plot(fpr1, tpr1, color='#C176DB', lw=2, label='TRA = %0.4f' % auc1)
	plt.plot(fpr2, tpr2, color='coral', lw=2, label='TRB = %0.4f' % auc2)
	plt.plot(fpr3, tpr3, color='steelblue', lw=2, label='Epitope = %0.4f' % auc3)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic Curve')
	plt.legend(loc="lower right")
	plt.savefig(os.path.join(output_dir,prefix+'_AntiSPencoder_test_label_roc.pdf'))


def plot_training_result(train_evaluation_df,val_evaluation_df,output_dir,prefix):
	plt.figure(None,(5,5))
	plt.title('Training loss')
	plt.plot(train_evaluation_df['loss'])
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(output_dir,prefix+"_AntiSPencoder_train_loss_epoch_scatter.pdf"))

	plt.figure(None,(5,5))
	plt.title('Training mask loss')
	plt.plot(train_evaluation_df['mask_loss'])
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(output_dir,prefix+"_AntiSPencoder_train_mask_loss_epoch_scatter.pdf"))

	plt.figure(None,(5,5))
	plt.title('Training class loss')
	plt.plot(train_evaluation_df['class_loss'])
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(output_dir,prefix+"_AntiSPencoder_train_class_loss_epoch_scatter.pdf"))
	
	plt.figure(None,(5,5))
	plt.title('Training accuracy1')
	plt.plot(train_evaluation_df['accuracy1'])
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	#plt.show()
	plt.savefig(os.path.join(output_dir,prefix+"_AntiSPencoder_train_accuracy1_epoch_scatter.pdf"))

	
	f, ax = plt.subplots(figsize = (10, 3))
	sns.violinplot(x="epoch",y="loss",data=val_evaluation_df)
	save_pdf_file = os.path.join(output_dir,prefix+'_AntiSPencoder_validation_loss_epoch_batch_violin.pdf')
	f.savefig(save_pdf_file,format='pdf', transparent=True,bbox_inches='tight')

	f, ax = plt.subplots(figsize = (10, 3))
	sns.violinplot(x="epoch",y="accuracy1",data=val_evaluation_df)
	save_pdf_file = os.path.join(output_dir,prefix+'_AntiSPencoder_validation_accuracy1_epoch_batch_violin.pdf')
	f.savefig(save_pdf_file,format='pdf', transparent=True,bbox_inches='tight')

	val_evaluation_df_mean = val_evaluation_df.groupby('epoch')[['loss','mask_loss','class_loss','accuracy1']].mean()
	val_evaluation_df_mean.reset_index(inplace=True)
	plt.figure(None,(5,5))
	plt.title('Validation loss')
	plt.plot(val_evaluation_df_mean['loss'])
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(output_dir,prefix+"_AntiSPencoder_validation_loss_epoch_scatter.pdf"))

	plt.figure(None,(5,5))
	plt.title('Validation mask loss')
	plt.plot(val_evaluation_df_mean['mask_loss'])
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(output_dir,prefix+"_AntiSPencoder_validation_mask_loss_epoch_scatter.pdf"))

	plt.figure(None,(5,5))
	plt.title('Validation class loss')
	plt.plot(val_evaluation_df_mean['class_loss'])
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(output_dir,prefix+"_AntiSPencoder_validation_class_loss_epoch_scatter.pdf"))
	
	plt.figure(None,(5,5))
	plt.title('Validation accuracy1')
	plt.plot(val_evaluation_df_mean['accuracy1'])
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	#plt.show()
	plt.savefig(os.path.join(output_dir,prefix+"_AntiSPencoder_validation_accuracy1_epoch_scatter.pdf"))


def create_dataset():
	torch.manual_seed(1996)
	# emb_data_file = "../data/train_embbeding_data_info_filter_C_TCRdb.txt"
	# merge_cdr3_epitope_info = pd.read_csv(emb_data_file,header=0,sep="\t")
	
	# dataset1 = PRETrainData(merge_cdr3_epitope_info['sequence'],merge_cdr3_epitope_info['flag'],
	# 	merge_cdr3_epitope_info['speci_flag'],merge_cdr3_epitope_info[["Autoimmune", "Cancer", "Pathogens","Epitope"]])
	# dataset2 = GetData(merge_cdr3_epitope_info['sequence'],merge_cdr3_epitope_info['flag'],
	# 	merge_cdr3_epitope_info['speci_flag'],merge_cdr3_epitope_info[["Autoimmune", "Cancer", "Pathogens","Epitope"]])

	# train_ratio = 0.9
	# val_ratio = 0.05
	# test_ratio = 0.05

	# # 计算各集合大小
	# train_size = int(train_ratio * len(dataset1))
	# val_size = int(val_ratio * len(dataset1))
	# test_size = len(dataset1) - train_size - val_size

	# generator = torch.Generator().manual_seed(1996)

	# train_dataset1, val_dataset1, test_dataset1 = random_split(dataset1, [train_size, val_size, test_size],generator=generator)
	# train_dataset2, val_dataset2, test_dataset2 = random_split(dataset2, [train_size, val_size, test_size],generator=generator)	

	# torch.save(train_dataset1,'../data/train_encoder_filter_C_TCRdb.pth')
	# torch.save(val_dataset2,'../data/val_encoder_filter_C_TCRdb.pth')
	# torch.save(test_dataset2,'../data/test_encoder_filter_C_TCRdb.pth')

	train_dataset = torch.load('../data/train_encoder_filter_C_TCRdb.pth')
	val_dataset = torch.load('../data/val_encoder_filter_C_TCRdb.pth')
	test_dataset = torch.load('../data/test_encoder_filter_C_TCRdb.pth')

	print(len(train_dataset))
	print(len(val_dataset))
	print(len(test_dataset))

	train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4)

	for i, data in enumerate(train_loader):
			
		input_sentence = data[0]
		input_label = data[1]

		print(input_sentence.size())
		print(input_label.size())
		break  # 只看第一个批次
	for i, data in enumerate(test_loader):
			
		input_sentence = data[0]
		

		print(input_sentence.size())
		break  # 只看第一个批次	
	return train_loader,val_loader,test_loader