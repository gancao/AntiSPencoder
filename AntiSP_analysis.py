import torch
import pandas as pd
import AntiSPencoder
import os

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

class_list = pd.DataFrame(['TRA','TRB','Antigen peptide'])


def get_huARdb():
	file = "../data/huARdbv2/huARdb_v2_GEX.CD8.all_genes_seurat_metadata.txt"
	metadata = pd.read_csv(file,sep="\t",header=0,index_col=0)
	tra_seqs = metadata['IR_VJ_1_junction_aa'].tolist()
	trb_seqs = metadata['IR_VDJ_1_junction_aa'].tolist()

	seqs = list(set(tra_seqs+trb_seqs))
	print("get_huARdb:",len(seqs))

	return seqs

def get_celltypist():
	file = "../data/celltypist/adata_TILC_TCR_onlyseq_seurat_metadata.txt"
	metadata = pd.read_csv(file,sep="\t",header=0,index_col=0)
	tra_seqs = metadata['IR_VJ_1_cdr3'].dropna().tolist()
	trb_seqs = metadata['IR_VDJ_1_cdr3'].dropna().tolist()
	seqs = list(set(tra_seqs+trb_seqs))
	seqs = list(set(seqs).difference(set(['nan'])))

	print("get_celltypist:",len(seqs))

	return seqs

def get_cdr3_epitope():
	file = "../data/TCR_epitope/merge_TCR_epitope_data_complete_cdr3_filter_rename_antigen_final.txt"
	info = pd.read_csv(file,sep="\t",header=0,index_col=0)
	epitope_ids = info['epitope'].dropna().tolist()
	tra_seqs = info['cdr3_tra'].dropna().tolist()
	trb_seqs = info['cdr3_trb'].dropna().tolist()
	seqs = list(set(epitope_ids+tra_seqs+trb_seqs))
	seqs = list(set(seqs).difference(set(['nan'])))
	return seqs

if __name__ == '__main__':
	model_dir = "../model"
	prefix = 'All'

	if torch.cuda.is_available():
		device = torch.device("cuda")  
	else:
		device = torch.device("cpu")

	print(device)
	
	encoder_path = os.path.join(model_dir, "filter_C_TCRdb_AntiSPencoder_checkpoint.ep29")
	pretrain = AntiSPencoder.TCREPbert(device=device)

	state_dict = torch.load(encoder_path, map_location=device)['model_state_dict'] # 加载文件
	pretrain.load_state_dict(state_dict)
	pretrain = pretrain.to(device)
	pretrain.eval()
	for param in pretrain.parameters():
		param.requires_grad = False
	pretrain.eval()

	#1.huARdb
	sequence = get_huARdb()
	encoder_save_file = "../analysis/huARdb_v2_GEX.CD8/tcr_encoder_info.txt"
	encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=256,num_workers=4)
	encoder_info.to_csv(encoder_save_file,sep="\t")
	embed_save_file = "../analysis/huARdb_v2_GEX.CD8/tcr_embedding_info.txt"
	embeddings_info.to_csv(embed_save_file,sep="\t")

	#2.celltypist
	sequence = get_celltypist()
	encoder_save_file = "../analysis/celltypist/tcr_encoder_info.txt"
	encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=256,num_workers=4)
	encoder_info.to_csv(encoder_save_file,sep="\t")
	embed_save_file = "../analysis/celltypist/tcr_embedding_info.txt"
	embeddings_info.to_csv(embed_save_file,sep="\t")

	#3.for train，valid,test encoder
	sequence = get_cdr3_epitope()
	encoder_save_file = "../analysis/tcr_antigen_db/tcr_epitope_complete_cdr3_filter_encoder_info.txt"
	encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=32,num_workers=4)
	encoder_info.to_csv(encoder_save_file,sep="\t")
	embed_save_file = "../analysis/tcr_antigen_db/tcr_epitope_complete_cdr3_filter_embedding_info.txt"
	embeddings_info.to_csv(embed_save_file,sep="\t")
