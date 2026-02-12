
cat("Start installing R packages...\n")

# 设置镜像
#options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

# 包列表
packages <- c(
  "data.table",
  "Matrix",
  "foreach",
  "doParallel",
  "stringdist",
  "stringr",
  "ggmsa",
  "ggseqlogo",
  "DECIPHER",
  "Biostrings",
  "wordcloud",
  "RColorBrewer",
  "doSNOW",
  "ggplot2",
  "dplyr",
  "ggpubr",
  "patchwork",
  "gtools"
)

# 安装函数
install_packages_safely <- function(pkg) {
  tryCatch(
    {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        cat("Install:", pkg, "\n")
        install.packages(pkg, dependencies = TRUE, quiet = TRUE)
        cat("✓", pkg, "Success\n")
      }else {
        cat("✓", pkg, "Installed\n")
      }
    }, error = function(e) {
      cat("Install:", pkg, "\n")
      tryCatch({
        if(!requireNamespace(pkg, quietly = TRUE)) {
          install.packages(pkg, type = "win.binary", dependencies = TRUE, quiet = TRUE)
          cat("✓", pkg, "Success\n")
        }
      },error = function(e) {
        cat("✗", pkg, "Error:", e$message, "\n")
      }) 
    }
  )
}

# 安装所有包
lapply(packages, install_packages_safely)

cat("R packages installed!\n")