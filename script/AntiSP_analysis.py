import torch
import pandas as pd
import AntiSPencoder
import os

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

def get_independent_test_list():
	file = "../data/TCR_epitope/test_embbeding_data_info_filter_C_independent_seq_rm_sample0.05.txt"
	info = pd.read_csv(file,sep="\t",header=0,index_col=0)
	seq_ids = info['sequence'].dropna().tolist()
	seqs = list(set(seq_ids))
	seqs = list(set(seqs).difference(set(['nan'])))

	# sampled_seqs = pandas_length_stratified_sample(seqs,sampling_ratio=0.1,seed=1996)

	return seqs

def get_independent_tcr_pmhc():
	file = "../data/TCR_epitope/test_TCR_epitope_data_info_filter_C_human_independent_pmhc_rm.txt"
	info = pd.read_csv(file,sep="\t",header=0,index_col=0)
	epitope_ids = info['epitope'].dropna().tolist()
	tra_seqs = info['cdr3_tra'].dropna().tolist()
	trb_seqs = info['cdr3_trb'].dropna().tolist()
	seqs = list(set(epitope_ids+tra_seqs+trb_seqs))
	seqs = list(set(seqs).difference(set(['nan'])))
	return seqs


if __name__ == '__main__':
	model_dir = "../model"
	if torch.cuda.is_available():
		device = torch.device("cuda")  
	else:
		device = torch.device("cpu")

	print(device)
	
	encoder_path = os.path.join(model_dir, "filter_C_TCRdb_AntiSPencoder_checkpoint.ep29")
	pretrain = AntiSPencoder.TCREPbert(device=device)

	state_dict = torch.load(encoder_path, map_location=device)['model_state_dict']
	pretrain.load_state_dict(state_dict)
	pretrain = pretrain.to(device)
	pretrain.eval()
	for param in pretrain.parameters():
		param.requires_grad = False
	pretrain.eval()

	#for TCR-epitope
	sequence = get_cdr3_epitope()
	encoder_save_file = "../analysis/tcr_antigen_db/tcr_epitope_complete_cdr3_filter_encoder_info.txt"
	encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=32,num_workers=4)
	embed_save_file = "../analysis/tcr_antigen_db/tcr_epitope_complete_cdr3_filter_embedding_info.txt"
	embeddings_info.to_csv(embed_save_file,sep="\t")

	#independent TCR dataset for AntiSPencoder
	sequence = get_independent_test_list()
	#for index,chunk in enumerate(sequence_chunks):
	encoder_save_file = "../analysis/independent_test/All_AntiSPencoder_independent_test_encoder_info.txt"
	encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=32,num_workers=4)
	encoder_info.to_csv(encoder_save_file,sep="\t")
	embed_save_file = "../analysis/independent_test/All_AntiSPencoder_independent_test_embedding_info.txt"
	embeddings_info.to_csv(embed_save_file,sep="\t")

	#independent TCR-epitope dataset for AntiSPencoder
	sequence = get_independent_tcr_pmhc()
	encoder_save_file = "../analysis/independent_tcr_antigen_db/tcr_epitope_human_independent_filter_encoder_info.txt"
	encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=32,num_workers=4)
	encoder_info.to_csv(encoder_save_file,sep="\t")
	embed_save_file = "../analysis/independent_tcr_antigen_db/tcr_epitope_human_independent_filter_embedding_info.txt"
	embeddings_info.to_csv(embed_save_file,sep="\t")

	#celltypist
	sequence = get_celltypist()
	encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=256,num_workers=4)
	embed_save_file = "../analysis/celltypist/tcr_embedding_info.txt"
	embeddings_info.to_csv(embed_save_file,sep="\t")

	#huARdb
	sequence = get_huARdb()
	encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=256,num_workers=4)
	embed_save_file = "../analysis/huARdb_v2_GEX.CD8/tcr_embedding_info.txt"
	embeddings_info.to_csv(embed_save_file,sep="\t")

