suppressPackageStartupMessages({
  library(Seurat)
  library(data.table)
})

rds_path <- "data/raw/human_glia_sn/GSE281069_20220516_seur_comb_subcluster_anno.rds.gz"
out_dir <- "data/interim/gse281069_pseudobulk"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_rds_auto <- function(path) {
  if (grepl("\\.gz$", path)) {
    con <- gzfile(path, "rb")
    on.exit(close(con))
    return(readRDS(con))
  }
  readRDS(path)
}

message("Reading Seurat object. This can take several minutes for the 4+ GB GSE281069 object...")
obj <- read_rds_auto(rds_path)
meta <- obj@meta.data
meta$cell_barcode <- rownames(meta)

fwrite(data.table(column = colnames(meta)), file.path(out_dir, "metadata_columns.csv"))
fwrite(as.data.table(meta[1:min(30, nrow(meta)), , drop = FALSE]), file.path(out_dir, "metadata_preview_first30.csv"))

sink(file.path(out_dir, "seurat_object_summary.txt"))
print(obj)
cat("\nAssays:\n")
print(Assays(obj))
cat("\nDefault assay:\n")
print(DefaultAssay(obj))
cat("\nMetadata dimensions:\n")
print(dim(meta))
cat("\nMetadata columns:\n")
print(colnames(meta))
cat("\nCandidate columns and examples:\n")
for (col in colnames(meta)) {
  if (grepl("condition|diagnosis|disease|status|group|dx|genotype|donor|sample|orig|library|cell|type|cluster|anno|region|brain|area|tissue|case|subject|patient", col, ignore.case = TRUE)) {
    vals <- unique(as.character(meta[[col]]))
    vals <- vals[!is.na(vals)]
    cat("\nCOLUMN:", col, "\n")
    cat("N unique:", length(vals), "\n")
    cat("Examples:", paste(head(vals, 30), collapse = " | "), "\n")
  }
}
sink()

message("Wrote:")
message(file.path(out_dir, "metadata_columns.csv"))
message(file.path(out_dir, "metadata_preview_first30.csv"))
message(file.path(out_dir, "seurat_object_summary.txt"))
