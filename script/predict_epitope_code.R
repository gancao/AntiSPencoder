library(data.table)
library(Matrix)
library(foreach)
library(doParallel)
library(stringdist)
library(stringr)

cosine_similarity <- function(A, B) {
  # 规范化矩阵行
  A_norm <- A / sqrt(rowSums(A^2))
  B_norm <- B / sqrt(rowSums(B^2))
  
  # 计算余弦相似度矩阵
  cosine_sim <- tcrossprod(A_norm, B_norm)  # 相当于 A_norm %*% t(B_norm)
  
  return(cosine_sim)
}

# 计算余弦距离（距离 = 1 - 相似度）
cosine_distance <- function(A, B) {
  1 - cosine_similarity(A, B)
}


sparse_batch_merge <- function(dist_matrix, metadata, threshold, 
                              target_column = "cdr3", chunksize = 1000,cores=10,aa_dist=3) {
  
  # 转换为data.table并设置键
  metadata_dt <- as.data.table(metadata)
  if(!target_column %in% names(metadata_dt)) {
    stop("Target column not found in metadata")
  }
  
  n_cols <- ncol(dist_matrix)
  total_chunks <- ceiling(n_cols / chunksize)
  all_results <- list()
  for(i in seq_len(total_chunks)) {
    start <- (i - 1) * chunksize + 1
    end <- min(i * chunksize, n_cols)
    
    if(i %% 10 == 0) cat(sprintf("Processing chunk %d/%d\n", i, total_chunks))
    
    # 处理当前块
    chunk_matrix <- dist_matrix[, start:end, drop = FALSE]
    positions <- which(chunk_matrix <= threshold, arr.ind = TRUE) #返回行和列的索引，有两列
    
    if(nrow(positions) > 0) {
      # 获取符合条件的行名
      matching_rownames <- unique(rownames(chunk_matrix)[positions[, 1]])
      # unique_pred_cdr3s <- unique(colnames(chunk_matrix)[positions[, 2]])
      
      # 在metadata中查找这些行名对应的所有行
      chunk_result <- metadata_dt[get(target_column) %in% matching_rownames]
      
      if(nrow(chunk_result) > 0) {
        # lv_dist_matrix <- stringdistmatrix(matching_rownames, unique_pred_cdr3s, 
        #                            method = "osa", 
        #                            nthread = cores)
    		# lv_positions <- which(lv_dist_matrix <= aa_dist, arr.ind = TRUE) #返回行和列的索引，有两列

    		pred_cdr3 = colnames(chunk_matrix)[positions[, 2]]
    		target_cdr3 = rownames(chunk_matrix)[positions[, 1]]

    		lv_distance <- stringdist(pred_cdr3,target_cdr3,method = "osa", nthread = cores)
    		
    		# 添加参考列信息
        ref_info <- data.table(
          pred_cdr3 = pred_cdr3,
          cdr3 = target_cdr3,
          distance = chunk_matrix[positions],
          lv_distance = lv_distance
        )

       	ref_info <- ref_info[lv_distance<=aa_dist]

				chunk_result <- metadata_dt[get(target_column) %in% unique(ref_info[, as.character(cdr3)])]

        chunk_result <- merge(ref_info, chunk_result, 
                             by.x = "cdr3", by.y = target_column, 
                             all.x = TRUE, allow.cartesian = TRUE)
        all_results[[i]] <- chunk_result
      }
    }
    
    if(i %% 20 == 0) gc()  # 定期垃圾回收
  }
  
  # 合并所有结果
  final_result <- rbindlist(all_results, use.names = TRUE, fill = TRUE)
  setDF(final_result)

  final_result <- final_result %>%
	  group_by(pred_cdr3) %>%
	  arrange(distance) %>%  # 默认升序，desc()表示降序
	  slice_head(n = 100) %>% # 取每组前100条
	  ungroup()
  return(final_result)
}


predict_epitope_by_min_distance <- function(pred_cdr3_emb_info,chain_info,cores=10,aa_dist=3){
	#cdr3_epitope_info <- read.delim(cdr3_epitope_path, header=TRUE, check.names=F)
	raw_cdr3_epitope_emb <- rownames(cdr3_epitope_emb_info)
	rownames(cdr3_epitope_emb_info) <- paste("DB-",rownames(cdr3_epitope_emb_info))

	names(raw_cdr3_epitope_emb) <- rownames(cdr3_epitope_emb_info)

	merge_emb_info <- rbind(cdr3_epitope_emb_info,pred_cdr3_emb_info)

	print("PCA analysis")
  set.seed(1996)
  pca_result <- irlba::prcomp_irlba(merge_emb_info,n=50)
  pca_df <- as.matrix(pca_result$x)
  rownames(pca_df) <- rownames(merge_emb_info)
  print("calculating cosine distance")
  # emb_distance <- 1-as.matrix(proxy::simil(pca_df[rownames(cdr3_epitope_emb_info),], pca_df[rownames(pred_cdr3_emb_info),], method="cosine"))
  # emb_distance <- apply(emb_distance,2,as.vector)
  emb_distance <- cosine_distance(pca_df[rownames(cdr3_epitope_emb_info),], pca_df[rownames(pred_cdr3_emb_info),])
  print(dim(emb_distance))
 
	rownames(emb_distance) <- raw_cdr3_epitope_emb[rownames(cdr3_epitope_emb_info)]
	colnames(emb_distance) <- rownames(pred_cdr3_emb_info)

  # norm_pca_df <- pca_df / sqrt(rowSums(pca_df^2))
  # # 计算余弦相似度矩??
  # similarity_pca_df <- tcrossprod(norm_pca_df)
  # distance_pca_df <- 1-similarity_pca_df
  # diag(distance_pca_df) <- 0

  chain_list <- c("TRA","TRB")
  epitope_list <- rownames(epitope_cutoff_info)
  pred_epitopes <- data.frame()
  for(i in 1:length(chain_list)){
  	c_chain <- chain_list[i]
  	print(c_chain)
  	c_pred_cdr3s <- rownames(pred_cdr3_emb_info)[chain_info==c_chain]  	

		if(c_chain=='TRA'){
			c_cutoff <- min(as.numeric(epitope_cutoff_info[,'tra_cutoff']))
			cdr3_epitope_info$cdr3 <- cdr3_epitope_info$cdr3_tra

		}else{
			c_cutoff <- min(as.numeric(epitope_cutoff_info[,'trb_cutoff']))
			cdr3_epitope_info$cdr3 <- cdr3_epitope_info$cdr3_trb
		}
		#c_cdr3_epitope_info <- cdr3_epitope_info[as.character(cdr3_epitope_info$epitope)==c_epitope,,drop=FALSE]

		c_db_cdr3s <- unique(as.character(cdr3_epitope_info$cdr3))
		print(setdiff(c_db_cdr3s,rownames(emb_distance)))
		print(setdiff(c_pred_cdr3s,colnames(emb_distance)))
		c_emb_distance <- emb_distance[c_db_cdr3s,c_pred_cdr3s,drop=FALSE]

	
		c_epitope_pred_info <- sparse_batch_merge(c_emb_distance, cdr3_epitope_info, c_cutoff,
			target_column = "cdr3",cores=cores,aa_dist=aa_dist)
		c_epitope_pred_info$pred_chain <- c_chain
		c_epitope_pred_info$cutoff <- c_cutoff

		pred_epitopes <- rbind(pred_epitopes,c_epitope_pred_info)
	}
	return(pred_epitopes)
}

library("gtools")
setwd(gtools::script_path())

cdr3_epitope_emb_file <- "../data/TCR_epitope/tcr_epitope_complete_cdr3_filter_embedding_info.txt"
epitope_cutoff_file <- "../data/TCR_epitope/AntiSPencoder_test_final_dist_cutoff_results_seed1996.txt"
cdr3_epitope_path <- "../data/TCR_epitope/merge_TCR_epitope_data_complete_cdr3_filter_rename_antigen_final.txt"
cdr3_epitope_info <- read.delim(cdr3_epitope_path,header=TRUE,row.names=1,check.names=FALSE)

cdr3_epitope_info <- subset(cdr3_epitope_info,select = -data_source)
cdr3_epitope_info <- subset(cdr3_epitope_info,select = -cell_type)
cdr3_epitope_info <- cdr3_epitope_info[!duplicated(cdr3_epitope_info),]
cdr3_epitope_info$ID <- rownames(cdr3_epitope_info)


cat(nrow(cdr3_epitope_info)," records")
cdr3_epitope_emb_info <- read.delim(cdr3_epitope_emb_file, header=TRUE, check.names=FALSE, row.names=1)
epitope_cutoff_info <- read.delim(epitope_cutoff_file,header=TRUE)
rownames(epitope_cutoff_info) <- as.character(epitope_cutoff_info$epitope)