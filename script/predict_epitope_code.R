library(data.table)
library(Matrix)
library(foreach)
library(doParallel)
library(stringdist)
library(stringr)
library(ggmsa)
library(ggseqlogo)
library(DECIPHER)
library(Biostrings)
library(wordcloud)
library(RColorBrewer)
library(doSNOW)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(patchwork)

custom_colors <- c(
  # 脂肪族
  'A' = '#1E90FF', 'V' = '#1E90FF', 'L' = '#1E90FF', 'I' = '#1E90FF',#- 道奇蓝
  
  # 芳香族
  'F' = '#8A2BE2', 'W' = '#8A2BE2', 'Y' = '#8A2BE2', #蓝紫色
  
  # 含硫氨基酸
  'C' = '#FFD700', 'M' = '#FFD700',# - 金色
  
  # 羟基氨基酸
  'S' = '#32CD32', 'T' = '#32CD32', #- 酸橙绿
  
  # 带电氨基酸
  'D' = '#FF4500', 'E' = '#FF4500',  # 酸性- 橙红色
  'K' = 'coral', 'R' = 'coral', 'H' = 'coral',  # 碱性
  
  # 酰胺类 Amide-containing (青色系)
  'N' = '#20B2AA',  # 天冬酰胺 - 浅海绿
  'Q' = '#40E0D0',  # 谷氨酰胺 - 绿松石色
  
  # 亚氨基酸 Imino acid (灰色系)
  'P' = '#808080',  # 脯氨酸 - 灰色
  
  # 特殊小分子 Special (粉色系)
  'G' = '#FF69B4'   # 甘氨酸 - 热粉色
)

# ?? ggmsa 创建颜色函数
ggmsa_color_scheme <- data.frame(names=names(custom_colors),color=custom_colors)

custom_scheme <- make_col_scheme(
  chars = names(custom_colors),
  cols = custom_colors
)

color_by_frequency <- function(words, freq) {
  colors <- character(length(words))
  breaks <- quantile(freq, probs = c(0, 0.25, 0.5, 0.75, 1))
  
  for (i in seq_along(words)) {
    if (freq[i] <= breaks[2]) {
      colors[i] <- "#E0E0E0"  # 灰色 - 低频
    } else if (freq[i] <= breaks[3]) {
      colors[i] <- "#4FC3F7"  # 浅蓝 - 中低频
    } else if (freq[i] <= breaks[4]) {
      colors[i] <- "#1976D2"  # 蓝色 - 中高频
    } else {
      colors[i] <- "#0D47A1"  # 深蓝 - 高频
    }
  }
  return(colors)
}

get_colors <- function(){
    # 20种对比鲜明的颜色（Hex 编码??
    contrast_colors_20 <- c(
      "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",  # 经典 Tableau 调色??
      "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",  # + 扩展??
      "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",  # 浅色变体
      "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"   # 柔和但可区分
    )
  colors <- subset(brewer.pal.info,category=='qual')
  colorqual <- c()
  for(i in nrow(colors):1) colorqual <- c(colorqual, brewer.pal(colors[i,'maxcolors'], name = rownames(colors)[i]) )
  colorqual <- c(rev(contrast_colors_20),colorqual)
  return(colorqual)
}

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

visualize_pred_info <- function(pred_epitopes,result_dir,prefix,metadata_info=NULL,cdr3_pair_info=NULL,ncores=10,plot_MSA=TRUE){
  static_dir <- "../static"
  copy_result <- file.copy(static_dir, result_dir, 
                        recursive = TRUE,  # 递归复制子目录
                        overwrite = TRUE)  # 覆盖已存在文件

  if(!is.null(cdr3_pair_info)){
      rm_cdr3_info <- cdr3_pair_info[!duplicated(cdr3_pair_info[,c("cdr3_tra","cdr3_trb")]),c("cdr3_tra","cdr3_trb")]
      rownames(rm_cdr3_info) <- paste(as.character(rm_cdr3_info$cdr3_tra),as.character(rm_cdr3_info$cdr3_trb))
    }else{
      print("cdr3_pair_info is NULL")
      tra_cdr3s <- unique(as.character(pred_epitopes[as.character(pred_epitopes$pred_chain)=='TRA',,drop=FALSE]$pred_cdr3))
      trb_cdr3s <- unique(as.character(pred_epitopes[as.character(pred_epitopes$pred_chain)=='TRB',,drop=FALSE]$pred_cdr3))

      rm_cdr3_info <- data.frame(
          cdr3_tra=c(tra_cdr3s,rep(NA,length(trb_cdr3s))),
          cdr3_trb=c(rep(NA,length(tra_cdr3s)),trb_cdr3s)
        )
      rm_cdr3_info <- rm_cdr3_info[(!is.na(rm_cdr3_info$cdr3_tra)) | (!is.na(rm_cdr3_info$cdr3_trb)),]
      
    }
    print(head(rm_cdr3_info))

    pred_epitopes <- pred_epitopes[as.character(pred_epitopes$pred_cdr3) %in% c(rm_cdr3_info$cdr3_tra,rm_cdr3_info$cdr3_trb),]
   # save_file <- paste(result_dir,"/",prefix,"_pred_epitopes_info_by_min_dist.txt",sep="")
   #  pred_epitopes <- read.delim(save_file,header=TRUE,check.names=FALSE)
   #  print(head(pred_epitopes))

   #step1: basic visualizations #step1: basic visualizations #step1: basic visualizations
    pred_epitopes$epitope_category <- as.character(pred_epitopes$epitope_category)
    pred_epitopes$epitope_name <- as.character(pred_epitopes$epitope_name)
    pred_epitopes$epitope_name[pred_epitopes$epitope_name == 'Melanoma:VV'] <- 'Melanoma'
    pred_epitopes$epitope_name[pred_epitopes$epitope_name == 'HCMV'] <- 'CMV'
    
    pred_species_lv_count <- plyr::count(pred_epitopes[, c("pred_chain","species","lv_distance")])
    colnames(pred_species_lv_count) <- c("pred_chain","species","lv_distance","count")
    pred_species_lv_count$lv_distance <- as.factor(pred_species_lv_count$lv_distance)
    
    pred_species_lv_count$species <- ifelse(as.character(pred_species_lv_count$species)=='HomoSapiens','HomoSapiens','Other')
    p <- ggplot(pred_species_lv_count, aes(x = lv_distance, y = count, fill = species)) +
      facet_wrap(~pred_chain,nrow=1,scales="free")+
      geom_bar(stat = "identity", width = 0.7) +
      scale_fill_manual(values=get_colors()[1:length(unique(pred_species_lv_count$species))]) +
      theme_classic() +
      labs(title = paste("Distribution of lv_distance across species"))
    save_plot_name <- paste(prefix,"_predict_lv_distance_species_ratio.pdf",sep="")
    ggsave(filename =  save_plot_name,device = "pdf",path = result_dir,width = 6,height = 2.5)

    #for HomoSapiens
    pred_epitopes <- pred_epitopes[as.character(pred_epitopes$species)=='HomoSapiens',,drop=FALSE]
    tra_pred_epitopes <- pred_epitopes[as.character(pred_epitopes$pred_chain)=='TRA',,drop=FALSE]
    trb_pred_epitopes <- pred_epitopes[as.character(pred_epitopes$pred_chain)=='TRB',,drop=FALSE]

    #which antigen epitopes?
    lv_cutoff <- c(0,1,2,3)
    p_antigen_list <- list()
    p_class_list <- list()

    group_colors <- c("#0D98DE","#F55F08","#FF9800","#7646BC","pink","grey")
    names(group_colors) <- c("Pathogens","Cancer","Autoimmune","HomoSP","Other","Unknown")

    for(i in 1:length(lv_cutoff)){
      c_cutoff <- lv_cutoff[i]
      c_pred_epitopes <- pred_epitopes[pred_epitopes$lv_distance<= c_cutoff,]
      
      pred_epitopes_count <- plyr::count(c_pred_epitopes[!duplicated(c_pred_epitopes[,c("pred_cdr3","pred_chain","cdr3_tra","cdr3_trb","epitope","epitope_name","MHC_class")]),
        c("pred_chain","epitope_name")])
      colnames(pred_epitopes_count) <- c("pred_chain","epitope_name","count")
      pred_epitopes_count$epitope_name <- as.character(pred_epitopes_count$epitope_name)
      pred_epitopes_count$epitope_name[rank(-1*pred_epitopes_count$count)>20] <- 'Other/Unknown'
      pred_epitopes_count$epitope_name[pred_epitopes_count$epitope_name=='Unknown'] <- 'Other/Unknown'

      pred_epitopes_count <- pred_epitopes_count %>%
        group_by(pred_chain) %>%
        mutate(
          percentage = count / sum(count) * 100,
          label = paste0(epitope_name, "\n", round(percentage, 2), "%"),
          # 计算标签位置（用于在扇形中间显示文字）
          ymax = cumsum(percentage),  # 累计最大值
          ymin = c(0, head(ymax, n = -1)),  # 累计最小值
          ypos = (ymax + ymin) / 2,  # 标签放在扇形中间
        )

      pred_epitopes_count <- as.data.frame(pred_epitopes_count)
      pred_epitopes_count$epitope_name <- reorder(pred_epitopes_count$epitope_name,-1*pred_epitopes_count$percentage)
      antigen_levels <- c(setdiff(levels(pred_epitopes_count$epitope_name),'Other/Unknown'),'Other/Unknown')
      pred_epitopes_count$epitope_name <- factor(pred_epitopes_count$epitope_name,levels=antigen_levels)

      pred_epitopes_count$label <- as.character(pred_epitopes_count$label)
      pred_epitopes_count$label[rank(-1*pred_epitopes_count$percentage)>10] <- ''
      p <- ggplot(pred_epitopes_count, aes(x = 2, y = percentage, fill = epitope_name)) +
        facet_wrap(~pred_chain,ncol=1,scales="free")+
        geom_bar(stat = "identity", width = 1) +
        coord_polar("y", start = 0) +
        #geom_text(aes(y = ypos, label = label), color = "black", size = 4) +
        scale_fill_manual(values=get_colors()[1:length(unique(pred_epitopes_count$epitope_name))]) +
        theme_void() +
        xlim(0.5, 2.5) +  # 这个设置创建了中间的空洞
        labs(title = paste("Distribution of predicted antigens for edit distance:",c_cutoff))
      p_antigen_list[[length(p_antigen_list)+1]] <- p
      # save_plot_name <- paste(prefix,"_predict_cdr3_lv_",c_cutoff,"_antigen_epitope_ratio.pdf",sep="")
      # ggsave(filename =  save_plot_name,device = "pdf",path = result_dir,width = 8.5,height = 3.5)

      pred_epitopes_count <- plyr::count(c_pred_epitopes[!duplicated(c_pred_epitopes[,c("pred_cdr3","pred_chain","cdr3_tra","cdr3_trb","epitope","epitope_name","MHC_class")]),
        c("pred_chain","epitope_category")])
      colnames(pred_epitopes_count) <- c("pred_chain","epitope_category","count")
      pred_epitopes_count$epitope_category <- as.character(pred_epitopes_count$epitope_category)
      pred_epitopes_count <- pred_epitopes_count %>%
        group_by(pred_chain) %>%
        mutate(
          percentage = count / sum(count) * 100,
          label = paste0(epitope_category, "\n", round(percentage, 2), "%"),
          # 计算标签位置（用于在扇形中间显示文字）
          ymax = cumsum(percentage),  # 累计最大值
          ymin = c(0, head(ymax, n = -1)),  # 累计最小值
          ypos = (ymax + ymin) / 2,  # 标签放在扇形中间
        )

      pred_epitopes_count <- as.data.frame(pred_epitopes_count)
      pred_epitopes_count$epitope_category <- reorder(pred_epitopes_count$epitope_category,-1*pred_epitopes_count$percentage)
      pred_epitopes_count$label <- as.character(pred_epitopes_count$label)
      pred_epitopes_count$label[rank(-1*pred_epitopes_count$percentage)>10] <- ''
      p <- ggplot(pred_epitopes_count, aes(x = 2, y = percentage, fill = epitope_category)) +
        facet_wrap(~pred_chain,ncol=1,scales="free")+
        geom_bar(stat = "identity", width = 1) +
        coord_polar("y", start = 0) +
        #geom_text(aes(y = ypos, label = label), color = "black", size = 4) +
        scale_fill_manual(values=group_colors) +
        theme_void() +
        xlim(0.5, 2.5) +  # 这个设置创建了中间的空洞
        labs(title = paste("Distribution of predicted antigens for edit distance:",c_cutoff))
      p_class_list[[length(p_class_list)+1]] <- p
      # save_plot_name <- paste(prefix,"_predict_cdr3_lv_",c_cutoff,"_antigen_epitope_category_ratio.pdf",sep="")
      # ggsave(filename =  save_plot_name,device = "pdf",path = result_dir,width = 8.5,height = 3.5)
    }

    save_plot_name <- paste(prefix,"_predict_cdr3_top_antigen_category.pdf",sep="")
    do.call(ggarrange, c(p_antigen_list,ncol=2,nrow=2))
    ggsave(filename =  save_plot_name,device = "pdf",path = result_dir,width = 16,height = 7)

    save_plot_name <- paste(prefix,"_predict_cdr3_top_antigen.pdf",sep="")
    do.call(ggarrange, c(p_class_list,ncol=2,nrow=2))
    ggsave(filename =  save_plot_name,device = "pdf",path = result_dir,width = 16,height = 7)

    #antigen specificity?
    lv_cutoff <- c(0,1,2,3)
    p_list <- list()
    for(i in 1:length(lv_cutoff)){
      c_cutoff <- lv_cutoff[i]
      c_pred_epitopes <- pred_epitopes[pred_epitopes$lv_distance<= c_cutoff,]
      c_pred_epitopes_count <- plyr::count(c_pred_epitopes[!duplicated(c_pred_epitopes[,c("pred_cdr3","pred_chain","epitope","MHC_class")]),
      c("pred_chain","pred_cdr3")])
      colnames(c_pred_epitopes_count) <- c("pred_chain","pred_cdr3","count")
      c_pred_epitopes_count$specificity <- as.character(paste("A",c_pred_epitopes_count$count,sep=""))
      c_pred_epitopes_count$specificity[c_pred_epitopes_count$count>5] <- "A>5" # ifelse(c_pred_epitopes_count$count>1,'polyspecific','monospecific')
      c_pred_epitopes_count_count <- plyr::count(c_pred_epitopes_count[,c("pred_chain","specificity")])
      colnames(c_pred_epitopes_count_count) <- c("pred_chain","specificity","count")
      c_pred_epitopes_count_count <- c_pred_epitopes_count_count %>%
        group_by(pred_chain) %>%
        mutate(
          percentage = count / sum(count) * 100,
          label = paste0(specificity, "\n", round(percentage, 1), "%"),
          # 计算标签位置（用于在扇形中间显示文字）
          ymax = cumsum(percentage),  # 累计最大值
          ymin = c(0, head(ymax, n = -1)),  # 累计最小值
          ypos = (ymax + ymin) / 2,  # 标签放在扇形中间
        )
      c_pred_epitopes_count_count <- as.data.frame(c_pred_epitopes_count_count)
      c_pred_epitopes_count_count$specificity <- factor(c_pred_epitopes_count_count$specificity,levels=c("A1","A2","A3","A4","A5","A>5"))
      p <- ggplot(c_pred_epitopes_count_count, aes(x = "", y = percentage, fill = specificity)) +
        facet_wrap(~pred_chain,nrow=2,scales="free")+
        geom_bar(stat = "identity", width = 1) +
        coord_polar("y", start = 0) +
        #geom_text(aes(y = ypos, label = label), color = "white") +
        scale_fill_brewer(palette = "Set2") +
        theme_void() +
        labs(title = paste("TCR specificity for lv:",c_cutoff,nrow(c_pred_epitopes_count)))

      p_list[[length(p_list)+1]] <- p

    }

    save_plot_name <- paste(prefix,"_predict_cdr3_antigen_specificity_pMHC.pdf",sep="")
    do.call(ggarrange, c(p_list,ncol=4,nrow=1))
    ggsave(filename =  save_plot_name,device = "pdf",path = result_dir,width = 10.5,height = 4)

    # #for HomoSapiens
    # ##find pair information
    
    print('calculating best antigen')
    cl <- makeCluster(ncores)
    registerDoSNOW(cl)
    pb <- txtProgressBar(max = nrow(rm_cdr3_info), style = 3)
    progress <- function(n) setTxtProgressBar(pb, n)
    opts <- list(progress = progress)

    antigen_info <- foreach::foreach(i=1:nrow(rm_cdr3_info),.combine=rbind,.inorder=T,.packages =c("dplyr"),
      .options.snow = opts) %dopar% {
      #antigen_info <- apply(rm_cdr3_info[,c("cdr3_tra","cdr3_trb")],1,function(x){
        x <- as.character(rm_cdr3_info[i,])
        flag_list = c()
        if(x[1] %in% as.character(tra_pred_epitopes$pred_cdr3)){
          flag_list <- c(flag_list,'TRA')
        }
        if(x[2] %in% as.character(trb_pred_epitopes$pred_cdr3)){
          flag_list <- c(flag_list,'TRB')
        }

        flag = paste(flag_list,collapse='/')

        if(length(flag_list)==2){
          c_tra_pred_cdr3 <- tra_pred_epitopes[as.character(tra_pred_epitopes$pred_cdr3) %in% x[1],,drop=FALSE]
          c_tra_pred_cdr3 <- c_tra_pred_cdr3 %>%
            mutate(
              tra_count=length(unique(cdr3_tra)),
              trb_count=0)  %>% ungroup()
          c_tra_count = c_tra_pred_cdr3$tra_count[1]

          c_trb_pred_cdr3 <- trb_pred_epitopes[as.character(trb_pred_epitopes$pred_cdr3) %in% x[2],,drop=FALSE]
          c_trb_pred_cdr3 <- c_trb_pred_cdr3 %>%
            mutate(
              tra_count=0,
              trb_count=length(unique(cdr3_trb)))  %>% ungroup()

          c_trb_count = c_trb_pred_cdr3$trb_count[1]
          
          c_pred_cdr3 <- rbind(c_tra_pred_cdr3,c_trb_pred_cdr3)
          
          c_pred_cdr3 <- c_pred_cdr3 %>%
            mutate(total_epitope_name=paste(unique(epitope_name),collapse=','),
              total_epitope_category=paste(unique(epitope_category),collapse=','),
              tra_count=c_tra_count,
              trb_count=c_trb_count)  %>%

            group_by(pred_chain,epitope_name) %>%
            mutate(score1=min(lv_distance),score2=min(distance))  %>% ungroup() %>%

            group_by(epitope_name) %>%
            mutate(mean_score1=max(score1),mean_score2=max(score2))

          c_pred_cdr3 <- as.data.frame(c_pred_cdr3)
          c_pred_cdr3 <- c_pred_cdr3[!duplicated(c_pred_cdr3[,c("epitope_name")]),,drop=FALSE]
          c_pred_antigen <- c_pred_cdr3[order(c_pred_cdr3$mean_score1,c_pred_cdr3$mean_score2),,drop=FALSE]

          if(x[2] %in% c_tra_pred_cdr3$cdr3_trb){
            flag = paste(flag_list,collapse='-')
          }
          if(x[1] %in% c_trb_pred_cdr3$cdr3_tra){
            flag = paste(flag_list,collapse='-')
          }

          return(c(as.character(c_pred_antigen$epitope_name)[1],
            as.character(c_pred_antigen$total_epitope_name)[1],
            as.character(c_pred_antigen$total_epitope_category)[1],

            c_pred_antigen$tra_count[1],
            c_pred_antigen$trb_count[1],
            as.numeric(c_pred_antigen$mean_score1)[1],
            as.numeric(c_pred_antigen$mean_score2)[1],
            as.character(c_pred_antigen$epitope_category)[1],
            as.character(c_pred_antigen$species)[1],
            flag
          ))
        }else{
          c_tra_pred_cdr3 <- tra_pred_epitopes[as.character(tra_pred_epitopes$pred_cdr3) %in% x[1],,drop=FALSE]
          c_tra_pred_cdr3 <- c_tra_pred_cdr3 %>%
            group_by(pred_cdr3) %>%
            mutate(total_epitope_name=paste(unique(epitope_name),collapse=','),
              total_epitope_category=paste(unique(epitope_category),collapse=','),
              tra_count=length(unique(cdr3_tra)),
              trb_count=0)  %>% ungroup()
              
          c_trb_pred_cdr3 <- trb_pred_epitopes[as.character(trb_pred_epitopes$pred_cdr3) %in% x[2],,drop=FALSE]
          c_trb_pred_cdr3 <- c_trb_pred_cdr3 %>%
            group_by(pred_cdr3) %>%
            mutate(total_epitope_name=paste(unique(epitope_name),collapse=','),
              total_epitope_category=paste(unique(epitope_category),collapse=','),
             tra_count=0,
              trb_count=length(unique(cdr3_trb)))  %>% ungroup()

          c_tra_pred_antigen <- c_tra_pred_cdr3[order(c_tra_pred_cdr3$lv_distance,c_tra_pred_cdr3$distance),,drop=FALSE]
          c_trb_pred_antigen <- c_trb_pred_cdr3[order(c_trb_pred_cdr3$lv_distance,c_trb_pred_cdr3$distance),,drop=FALSE]
          if(nrow(c_tra_pred_antigen)>0){         
            return(c(as.character(c_tra_pred_antigen$epitope_name)[1],
              as.character(c_tra_pred_antigen$total_epitope_name)[1],
              as.character(c_tra_pred_antigen$total_epitope_category)[1],

               c_tra_pred_antigen$tra_count[1],
              c_tra_pred_antigen$trb_count[1],
              as.numeric(c_tra_pred_antigen$lv_distance)[1],
              as.numeric(c_tra_pred_antigen$distance)[1],
              as.character(c_tra_pred_antigen$epitope_category)[1],
              as.character(c_tra_pred_antigen$species)[1],
              flag
            ))
          }
          if(nrow(c_trb_pred_antigen)>0){
            return(c(as.character(c_trb_pred_antigen$epitope_name)[1],
              as.character(c_trb_pred_antigen$total_epitope_name)[1],
              as.character(c_trb_pred_antigen$total_epitope_category)[1],              
              c_trb_pred_antigen$tra_count[1],
              c_trb_pred_antigen$trb_count[1],
              as.numeric(c_trb_pred_antigen$lv_distance)[1],
              as.numeric(c_trb_pred_antigen$distance)[1],
              as.character(c_trb_pred_antigen$epitope_category)[1],
              as.character(c_trb_pred_antigen$species)[1],
              flag
            ))
          }
          return(c('','','',0,0,NA,NA,'','',flag))
        }
    }
    close(pb)
    stopCluster(cl)

    print('finished')


    antigen_info <- as.data.frame(as.matrix(antigen_info))
    rownames(antigen_info) <- NULL
    colnames(antigen_info) <- c("antigen","total_antigen","total_category","tra_count","trb_count","lv_dist","emb_dist","epitope_category","cdr3_species","flag")

    rm_cdr3_info <- cbind(rm_cdr3_info,antigen_info)
    save_file <- paste(result_dir,"/",prefix,"_predict_human_pair_tcr_best_antigen_info.txt",sep="")
    write.table(rm_cdr3_info,save_file,quote=FALSE,sep="\t")

    pred_cdr3_info <- rm_cdr3_info[as.character(rm_cdr3_info$flag)!='',]

    pred_cdr3_info_count <- plyr::count(pred_cdr3_info$flag)
    colnames(pred_cdr3_info_count) <- c("chain","count")
    pred_cdr3_info_count$ratio <- pred_cdr3_info_count$count/nrow(rm_cdr3_info)   
    pred_cdr3_info_count$chain <- factor(pred_cdr3_info_count$chain,levels=c("TRA","TRB","TRA/TRB","TRA-TRB")) 
    #count ratio
    p <- ggplot(data=pred_cdr3_info_count,aes(x=chain,y=ratio))
    p <- p + geom_bar(stat="identity",fill='coral',width=0.7)
    p <- p + geom_text(aes(label=count))
    p <- p + theme_bw()
    p <- p + ggtitle(paste("The ratio of predicted CDR3s for all pairs",sep=""))
    p <- p + theme(axis.text.x = element_text(angle = 30, hjust = 1),
                            plot.title = element_text(hjust = 0.5),
                            strip.background = element_blank()
                            )
    save_plot_name <- paste(prefix,"_predict_cdr3_ratio.pdf",sep="")
    ggsave(filename =  save_plot_name,device = "pdf",path = result_dir,width = 3,height = 3)


  if(is.null(metadata_info)){
    rm_pred_cdr3_info <- pred_epitopes[!duplicated(pred_epitopes[,c("pred_cdr3","pred_chain")]),]
    metadata_info <- as.character(rm_pred_cdr3_info$pred_cdr3)
    names(metadata_info) <- paste(rm_pred_cdr3_info$pred_chain,1:nrow(rm_pred_cdr3_info))
  }
    

  #step2: visualizations in web page
  #(1): table
  #(2): multiple sequence alignment
  save_html_file <- paste(result_dir,"/",prefix,"_predict_TCR-pMHC_visualizations.html",sep="")

  cat(r"(
  }
<!DOCTYPE html>
<html>
 <head>
  <meta http-equiv="Content-Type" content="textml; charset=utf-8" />
  <meta name="Generator" content="EditPlus">
  <meta name="Author" content="GanRui">
  <meta name="Keywords" content="TCR,CDR3,antigen">
  <meta name="Description" content="TCR-pMHC interaction">
  <meta name="renderer" content="webkit" /> 
  <link rel="stylesheet" type="text/css" href="static/css/jquery.dataTables.min.css">
  <link rel="stylesheet" type="text/css" href="static/css/buttons.dataTables.min.css">
  <link rel="stylesheet" href="static/css/bootstrap.min.css">
  <link rel="stylesheet" href="static/fonts/bootstrap-icons.css">
  <link rel="stylesheet" href="static/css/tabulator.min.css" rel="stylesheet">
  <link rel="stylesheet" href="static/css/scroller.dataTables.min.css">

</head>
<script src="static/js/jquery.min.js"></script>
<script type="text/javascript" src="static/js/tabulator.min.js"></script>
<script type="text/javascript" language="javascript" src="static/js/bootstrap.min.js"></script>
<script type="text/javascript" language="javascript" src="static/js/datatable/jquery.js"></script>
<script type="text/javascript" language="javascript" src="static/js/datatable/jquery.dataTables.js"></script>
<script type="text/javascript" language="javascript" src="static/js/datatable/dataTables.buttons.min.js"></script>
<script type="text/javascript" language="javascript" src="static/js/datatable/jszip.min.js"></script>
<script type="text/javascript" language="javascript" src="static/js/datatable/pdfmake.min.js"></script>
<script type="text/javascript" language="javascript" src="static/js/datatable/vfs_fonts.js"></script>
<script type="text/javascript" language="javascript" src="static/js/datatable/buttons.html5.min.js"></script>
<script type="text/javascript" language="javascript" src="static/js/datatable/bootstrap.js"></script>
<script type="text/javascript" language="javascript" src="static/js/ui-bootstrap-1.3.0.min.js"></script>
<script type="text/javascript" language="javascript" src="static/js/dataTables.scroller.min.js"></script>

<script>
  $(function(){
    $('.display').DataTable({
          paging: true,
          
          //scrollCollapse: true,
          //scroller: true,
          responsive: true,
            
  

          deferRender: true, // 延迟渲染
          searching: true,
          language: {
              aria: {
                  sortAscending: ": activate to sort column ascending",
                  sortDescending: ": activate to sort column descending"
              },
              emptyTable: "No relative records",
              info: "_START_ - _END_ rows/total _TOTAL_ rows",
              infoEmpty: "No records",
              infoFiltered: "1 to _MAX_ rows",
              zeroRecords: "No records",
              lengthMenu: "_MENU_ rows for every page&nbsp&nbsp"
          },
         
          // 默认每页显示10行
          pageLength: 10,
          deferLoading: 0,
          "order": [[1,'asc'],[2,"desc"],[3,'desc']],
          dom: 'Bfrtip',
          buttons: [
              'copyHtml5',
              'excelHtml5',
              'csvHtml5'
          ]
    });


  }) 
</script>

<body>
  <div style="margin:50px;font-size: 12px;text-align:center;">
    <h1 style="color:orange;font-weight:bold;">Predictions of antigen-specificity via known TCR-pMHC pairs</h1>
    )",file = save_html_file)

  # cat(sprintf('<embed id="pdf-embed" src="%s_all_antigen_name_wordcloud.pdf" style="width:80%%;height:600px;text-align: center;" type="application/pdf">
  #    <div class="panel-group" id="accordion"
  #   ',prefix
  #   ),file = save_html_file, append = TRUE) 

  cat(sprintf('<img src="%s_all_antigen_name_wordcloud.png" style="width:50%%;text-align: center;">
    <div class="panel-group" id="accordion">
    ',prefix
    ),file = save_html_file, append = TRUE) 

  if(!is.null(cdr3_pair_info)){
    cat(sprintf('
        <div class="panel panel-default">
            <div class="panel-heading">
              <h4 class="panel-title">                
                <a style="font-weight: bold;text-align:left;">Summary of predicted antigens for input CDR3s</a>
                <a class="accordion-toggle collapsed" data-toggle="collapse" href="#collapse0">
                <span class="pull-right glyphicon glyphicon-chevron-down"></span></a>
              </h4>
            </div>
            <div id="collapse0" class="panel-collapse collapse in">
                <div class="panel-body">
                        <embed id="pdf-embed" src="%s_predict_cdr3_ratio.pdf" 
                        style="height:300px;text-align: center;" type="application/pdf"> 
                </div>
            </div>
        </div>
        ',prefix
    ),file = save_html_file, append = TRUE)
  }

  cat(sprintf('
        <div class="panel panel-default">
            <div class="panel-heading">
              <h4 class="panel-title">                
                <a style="font-weight: bold;text-align:left;">Summary of predicted antigens for input CDR3s</a>
                <a class="accordion-toggle collapsed" data-toggle="collapse" href="#collapse1">
                <span class="pull-right glyphicon glyphicon-chevron-down"></span></a>
              </h4>
            </div>
            <div id="collapse1" class="panel-collapse collapse in">
                <div class="panel-body">
                        <embed id="pdf-embed" src="%s_predict_lv_distance_species_ratio.pdf" 
                        style="width:100%%;height:350px;text-align: center;" type="application/pdf"> 
                </div>
            </div>
        </div>

        <div style="height:30px;"></div>

        <div class="panel panel-default">
            <div class="panel-heading">
              <h4 class="panel-title">
                <a style="font-weight: bold;text-align:left;">Top predicted antigens for input CDR3s</a>
                <a class="accordion-toggle collapsed" data-toggle="collapse" href="#collapse2">
                <span class="pull-right glyphicon glyphicon-chevron-down"></span></a>
              </h4>
            </div>
            <div id="collapse2" class="panel-collapse collapse in">
                <div class="panel-body">                        

                        <embed id="pdf-embed" src="%s_predict_cdr3_top_antigen_category.pdf" 
                        style="width:100%%;height:550px;text-align: center;" type="application/pdf"> 

                        <embed id="pdf-embed" src="%s_predict_cdr3_top_antigen.pdf" 
                        style="width:100%%;height:550px;text-align: center;" type="application/pdf"> 
                        
                </div>
            </div>
        </div>

        <div style="height:30px;"></div>

        <div class="panel panel-default">
            <div class="panel-heading">
              <h4 class="panel-title">                
                <a style="font-weight: bold;text-align:left;">Distribution of antigen-specificity for input CDR3s</a>
                <a class="accordion-toggle collapsed" data-toggle="collapse" href="#collapse3">
                <span class="pull-right glyphicon glyphicon-chevron-down"></span></a>
              </h4>
            </div>
            <div id="collapse3" class="panel-collapse collapse in">
                <div class="panel-body">                                             
                        <embed id="pdf-embed" src="%s_predict_cdr3_antigen_specificity_pMHC.pdf" 
                        style="width:100%%;height:400px;text-align: center;" type="application/pdf"> 
                </div>
            </div>
        </div>
    </div>
                            
    ',prefix,prefix,prefix,prefix
    ),file = save_html_file, append = TRUE)


  cat(r"(
  <table id="my-table" class="display" class="table table-striped table-hover table-bordered" style="border:0px;"> 
    <thead>
      <tr style="text-align:center;">
        <th>Query_TRA_CDR3</th>
        <th>Query_TRB_CDR3</th>
        <th>Matched_TRA_count</th>
        <th>Matched_TRB_count</th>
        <th>Target_antigen</th>
        <th>Target_category</th>
        <th>Target_best_antigen</th>
        <th>Best_edit_distance</th>
        <th>Detail</th>
        <th>Visualization</th>
      </tr>
    </thead>
    <tbody>

      )",file = save_html_file, append = TRUE)  


  pred_epitopes <- pred_epitopes[,c("pred_cdr3","pred_chain","cdr3","cdr3_tra","cdr3_trb","epitope","epitope_name","MHC_class","MHC_allele","lv_distance")]
  pred_epitopes <- pred_epitopes[!duplicated(pred_epitopes),]

  epitope_count <- plyr::count(pred_epitopes$epitope_name)
  colnames(epitope_count) <- c("word","freq")
  word_colors <- color_by_frequency(epitope_count$word, epitope_count$freq)
  save_pdf_file <- paste(result_dir,"/",prefix,"_all_antigen_name_wordcloud.png",sep="")
  png(save_pdf_file,width=800,height=800,res=350)
  wordcloud(epitope_count$word, epitope_count$freq,
                 scale = c(5, 0.6),  
                 colors = word_colors,
                 random.order = FALSE,
                 random.color = FALSE,
                 #size = 1,          # 字体大小
                 #shape = 'circle',  # 形状：circle, cardioid, diamond等
                 #color = 'random-dark',  # 颜色方案
                 backgroundColor = "white")
  #plot(p)
  dev.off()


  show_result_dir <- paste(result_dir,"/show",sep="")
  dir.create(show_result_dir)

  pred_epitopes <- pred_epitopes[order(pred_epitopes$pred_chain),]
  
 

  for(i in 1:nrow(pred_cdr3_info)){
      c_best_info <- pred_cdr3_info[i,,drop=FALSE]
      c_tra_cdr3 <- c_best_info$cdr3_tra[1]
      c_trb_cdr3 <- c_best_info$cdr3_trb[1]

      c_pred_epitopes <- pred_epitopes[as.character(pred_epitopes$pred_cdr3) %in% c(c_tra_cdr3,c_trb_cdr3),,drop=FALSE]
      c_table_pred_epitopes <- c_pred_epitopes[!duplicated(c_pred_epitopes),,drop=FALSE]
      # print(nrow(c_table_pred_epitopes))
      # print(head(c_table_pred_epitopes))

      c_table_info <- sprintf('
            <tr style="text-align:center;">     
              <td>%s</td>
              <td>%s</td>
              <td>%s</td>
              <td>%s</td>
              <td style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word;">%s</td>
              <td style="word-wrap: break-word; word-break: break-word; overflow-wrap: break-word;">%s</td>
              <td>%s</td>
              <td>%s</td>
              <td>
                <button data-toggle="modal" data-target="#AntiSPencoder_%s" style="background-color: #AD2417;color:white;width:50px;height:25px;border-radius: 5px;">Detail</button>
                <div class="modal fade" id="AntiSPencoder_%s" tabindex="-1" role="dialog" aria-labelledby="myModalLabel" aria-hidden="true">
                  <div class="modal-dialog" style="width:100%%;text-align:center;">
                    <div class="modal-content" style="width:100%%;text-align:center;">
                      <div class="modal-header">
                        <button type="button" class="close" data-dismiss="modal" aria-hidden="true"></button>
                        <h4 class="modal-title" id="myModalLabel">Matched TCR-pMHC information</h4>
                      </div>
                      <div class="modal-body">                            
                        <table class="display" class="table table-striped table-hover table-bordered" style="text-align: center;">
                          <thead>
                            <tr>
                              <th>Query_CDR3</th>
                              <th>Query_chain</th>
                              <th>Target_CDR3</th>
                              <th>Target_TRA_CDR3</th>
                              <th>Target_TRB_CDR3</th>
                              <th>Target_antigen</th>
                              <th>Target_epitope</th>                             
                              <th>Target_MHC</th>
                              <th>Target_allele</th>
                              <th>Edit_distance</th>
                            </tr>
                          </thead>
                          <tbody>
                              ',
                              ifelse(is.na(c_tra_cdr3),'',c_tra_cdr3),
                              ifelse(is.na(c_trb_cdr3),'',c_trb_cdr3),
                              c_best_info$tra_count[1],
                              c_best_info$trb_count[1],
                              c_best_info$total_antigen[1],
                              c_best_info$total_category[1],
                              c_best_info$antigen[1],
                              c_best_info$lv_dist[1],
                              i,
                              i
                            )
      cat(c_table_info,file = save_html_file, append = TRUE)   

      for(j in 1:nrow(c_table_pred_epitopes)){
        c_pred_info <- c_table_pred_epitopes[j,,drop=FALSE]
        c_pred_cdr3 <- c_pred_info$pred_cdr3[1]
        c_cdr3_name <- as.character(metadata_info[c_pred_cdr3])[1]
        c_table_info <- sprintf('
                            <tr style="text-align:center;">   
                              <td>%s</td>  
                              <td>%s</td>
                              <td>%s</td>
                              <td>%s</td>
                              <td>%s</td>
                              <td>%s</td>
                              <td>%s</td>
                              <td>%s</td>
                              <td>%s</td>
                              <td>%s</td>
                            </tr>
            ',
            c_pred_info$pred_cdr3[1],
            c_pred_info$pred_chain[1],
            c_pred_info$cdr3[1],            
            c_pred_info$cdr3_tra[1],
            c_pred_info$cdr3_trb[1],
            c_pred_info$epitope_name[1],
            c_pred_info$epitope[1],           
            c_pred_info$MHC_class[1],
            c_pred_info$MHC_allele[1],
            c_pred_info$lv_distance[1]
          )
          cat(c_table_info,file = save_html_file, append = TRUE)  
            
      }
      cat("
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              </td>
            ",file = save_html_file, append = TRUE)

      c_plot_pred_epitopes <- c_pred_epitopes[!duplicated(c_pred_epitopes[,c("cdr3","epitope_name","MHC_class","MHC_allele","epitope")]),,drop=FALSE]
     
      c_tra_plot_pred_epitopes <- c_plot_pred_epitopes[as.character(c_plot_pred_epitopes$pred_cdr3)%in%c_tra_cdr3,,drop=FALSE]
      c_trb_plot_pred_epitopes <- c_plot_pred_epitopes[as.character(c_plot_pred_epitopes$pred_cdr3)%in%c_trb_cdr3,,drop=FALSE]

      c_table_info <- '
            
              <td>/</td>
            </tr>'
      if(nrow(c_tra_plot_pred_epitopes)>1 & nrow(c_trb_plot_pred_epitopes)>1){ 
          
        c_table_info <- sprintf('
           
              <td>
                <a href="show/%s_multiple_alignment.pdf" target="_blank">
                  <button style="background-color: #AD2417;color:white;width:50px;height:25px;border-radius: 5px;">TRA</button>
                </a>
                <a href="show/%s_multiple_alignment.pdf" target="_blank">
                  <button style="background-color: #AD2417;color:white;width:50px;height:25px;border-radius: 5px;">TRB</button>
                </a>
              </td>
            </tr>',c_tra_cdr3,c_trb_cdr3)
      }
      if(nrow(c_tra_plot_pred_epitopes)>1 & nrow(c_trb_plot_pred_epitopes)<=1){ 
          
        c_table_info <- sprintf('
           
              <td>
                <a href="show/%s_multiple_alignment.pdf" target="_blank">
                  <button style="background-color: #AD2417;color:white;width:50px;height:25px;border-radius: 5px;">TRA</button>
                </a>
               
              </td>
            </tr>',c_tra_cdr3)
      }
      if(nrow(c_tra_plot_pred_epitopes)<=1 & nrow(c_trb_plot_pred_epitopes)>1){ 
          
        c_table_info <- sprintf('
              <td>
                <a href="show/%s_multiple_alignment.pdf" target="_blank">
                  <button style="background-color: #AD2417;color:white;width:50px;height:25px;border-radius: 5px;">TRB</button>
                </a>
               
              </td>
            </tr>',c_trb_cdr3)
      }
      cat(c_table_info,file = save_html_file, append = TRUE)
  }

  cat(r"(   
        </tbody>
      </table>
    </div>
  </body>
</html>    
    )",file = save_html_file, append = TRUE)

  if(!plot_MSA){
    return('finished')
  }
  print("plot MSA results")

  all_pred_cdr3s <- unique(as.character(pred_epitopes$pred_cdr3))
  print(length(all_pred_cdr3s))

  cl <- makeCluster(ncores)
  registerDoSNOW(cl)
  pb <- txtProgressBar(max = length(all_pred_cdr3s), style = 3)
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress = progress)

  foreach::foreach(i=1:length(all_pred_cdr3s),.combine=rbind,.inorder=T,.export =c("ggmsa_color_scheme","custom_scheme"),
    .packages = c("ggplot2","DECIPHER","ggmsa","ggseqlogo","Biostrings","patchwork"),.options.snow = opts) %dopar% {
    #for(i in 1:length(all_pred_cdr3s)) {   
      c_cdr3 <- all_pred_cdr3s[i]
      c_cdr3_name <- as.character(metadata_info[c_cdr3])[1]
      
      c_pred_epitopes <- pred_epitopes[as.character(pred_epitopes$pred_cdr3)==c_cdr3,,drop=FALSE]
      c_plot_pred_epitopes <- c_pred_epitopes[!duplicated(c_pred_epitopes[,c("cdr3","epitope_name","MHC_class","MHC_allele","epitope")]),,drop=FALSE]

      c_pred_name <- paste(as.character(c_plot_pred_epitopes$MHC_class),as.character(c_plot_pred_epitopes$MHC_allele),as.character(c_plot_pred_epitopes$epitope_name),as.character(c_plot_pred_epitopes$epitope),sep='+')

      c_pred_name <- paste(c_pred_name,'_',1:length(c_pred_name),sep='')

      c_target_cdr3s <- c(c_cdr3,as.character(c_plot_pred_epitopes$cdr3))
      names(c_target_cdr3s) <- c(c_cdr3_name,c_pred_name)

      if(length(c_target_cdr3s)==2){
        return(NULL)
      }
      print(c_target_cdr3s)
      aa_stringset <- AAStringSet(c_target_cdr3s)
      alignment <- AlignSeqs(aa_stringset, 
              verbose = FALSE,
              # 更宽松的空位罚分，适应CDR3长度变异
              # 增加迭代以获得更好结??
              useStructures = TRUE,
              guideTree = NULL,
              iterations = 3,
              gapOpening = -15,
              gapExtension = -2,
              terminalGap = -10,
              refinements = 2)
      save_file <- paste(show_result_dir,"/",c_cdr3,"_aligned.fasta",sep="")
      writeXStringSet(alignment,file = save_file)
      # Sys.sleep(0.5)

      alignment <- readAAMultipleAlignment(save_file,format = "fasta")
      alignment_character <- as.character(alignment)

      p1 <- ggmsa(
        save_file,                  
        color = "Chemistry_AA",
        custom_color=ggmsa_color_scheme,
        # font = "helvetica",
        # char_width = 0.6,
        seq_name = TRUE,
        # consensus_views = TRUE,
        show.legend = TRUE
      ) +
        theme(
          axis.text.y = element_text(face = "bold"),
          plot.title = element_text(hjust = 0.5)
        ) +
        labs(title = paste(c_cdr3_name))

      p2 <- ggseqlogo(
          alignment_character,
          method = 'bits',
          seq_type = 'aa',
          #col_scheme = 'chemistry'
          col_scheme = custom_scheme
        ) + ggtitle("Sequence Logo")

      combined_plot <- p1 / p2 + plot_layout(heights = c(max(1.2,length(c_target_cdr3s)/7.5), 1))
      #combined_plot <- (p1 / p2) | (p3 / p4) + plot_layout(heights = c(max(1,length(c_tra_cdr3s)/8), 1))
      #combined_plot <- p1  | p3    
      save_plot_name <- paste(c_cdr3,"_multiple_alignment.pdf",sep="")
      ggsave(filename =  save_plot_name,device = "pdf",path = show_result_dir,width = 8,height = min(30,length(c_target_cdr3s)/3+2.5))
    
  }
  close(pb)
  stopCluster(cl)

}


library("gtools")
setwd(gtools::script_path())

# cdr3_epitope_emb_file <- "../data/TCR_epitope/tcr_epitope_complete_cdr3_filter_embedding_info.txt"
# epitope_cutoff_file <- "../data/TCR_epitope/AntiSPencoder_test_final_dist_cutoff_results_seed1996.txt"
# cdr3_epitope_path <- "../data/TCR_epitope/merge_TCR_epitope_data_complete_cdr3_filter_rename_antigen_final.txt"
# cdr3_epitope_info <- read.delim(cdr3_epitope_path,header=TRUE,row.names=1,check.names=FALSE)

# cdr3_epitope_info <- subset(cdr3_epitope_info,select = -data_source)
# cdr3_epitope_info <- subset(cdr3_epitope_info,select = -cell_type)
# cdr3_epitope_info <- cdr3_epitope_info[!duplicated(cdr3_epitope_info),]
# cdr3_epitope_info$ID <- rownames(cdr3_epitope_info)


# cat(nrow(cdr3_epitope_info)," records")
# cdr3_epitope_emb_info <- read.delim(cdr3_epitope_emb_file, header=TRUE, check.names=FALSE, row.names=1)
# epitope_cutoff_info <- read.delim(epitope_cutoff_file,header=TRUE)
# rownames(epitope_cutoff_info) <- as.character(epitope_cutoff_info$epitope)