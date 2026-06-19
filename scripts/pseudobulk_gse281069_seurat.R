suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
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

pick_col <- function(patterns, cols) {
  for (pat in patterns) {
    hits <- grep(pat, cols, ignore.case = TRUE, value = TRUE)
    if (length(hits) > 0) return(hits[1])
  }
  return(NA_character_)
}

std_condition <- function(x) {
  z <- tolower(as.character(x))
  ifelse(grepl("control|normal|wt|wild|healthy|ctrl", z) & !grepl("hd|huntington", z), "control",
         ifelse(grepl("hd|huntington|disease|case|mutant", z), "HD", "unknown"))
}

message("Reading Seurat object: ", rds_path)
obj <- read_rds_auto(rds_path)
meta <- obj@meta.data
meta$cell_barcode <- rownames(meta)
cols <- colnames(meta)

condition_col <- Sys.getenv("GSE281069_CONDITION_COL", unset = "")
donor_col <- Sys.getenv("GSE281069_DONOR_COL", unset = "")
sample_col <- Sys.getenv("GSE281069_SAMPLE_COL", unset = "")
celltype_col <- Sys.getenv("GSE281069_CELLTYPE_COL", unset = "")
region_col <- Sys.getenv("GSE281069_REGION_COL", unset = "")

if (condition_col == "") condition_col <- pick_col(c("condition", "diagnosis", "disease", "status", "group", "dx", "genotype"), cols)
if (donor_col == "") donor_col <- pick_col(c("donor", "individual", "subject", "patient", "case"), cols)
if (sample_col == "") sample_col <- pick_col(c("orig.ident", "sample", "library", "specimen", "gsm"), cols)
if (celltype_col == "") celltype_col <- pick_col(c("cell.?type", "subcluster", "annotation", "anno", "cluster", "class"), cols)
if (region_col == "") region_col <- pick_col(c("region", "brain", "area", "tissue"), cols)

mapping <- list(condition_col=condition_col, donor_col=donor_col, sample_col=sample_col, celltype_col=celltype_col, region_col=region_col)
print(mapping)

if (any(is.na(unlist(mapping)) | unlist(mapping) == "")) {
  fwrite(data.table(column = cols), file.path(out_dir, "metadata_columns.csv"))
  stop("Could not infer all metadata columns. Inspect metadata_columns.csv and rerun with GSE281069_*_COL env vars.")
}

meta$condition_std <- std_condition(meta[[condition_col]])
meta$donor_std <- as.character(meta[[donor_col]])
meta$sample_std <- as.character(meta[[sample_col]])
meta$celltype_std <- as.character(meta[[celltype_col]])
meta$region_std <- as.character(meta[[region_col]])
meta <- meta[meta$condition_std %in% c("HD", "control"), ]
message("Condition table:")
print(table(meta$condition_std, useNA = "ifany"))

assay_name <- DefaultAssay(obj)
counts <- tryCatch(
  GetAssayData(obj, assay = assay_name, slot = "counts"),
  error = function(e1) {
    tryCatch(GetAssayData(obj, assay = assay_name, layer = "counts"), error = function(e2) GetAssayData(obj, assay = assay_name, layer = "data"))
  }
)
common_cells <- intersect(colnames(counts), rownames(meta))
if (length(common_cells) < 1000) common_cells <- intersect(colnames(counts), meta$cell_barcode)
message("Matched cells: ", length(common_cells))
if (length(common_cells) < 1000) stop("Too few matched cells between Seurat counts and metadata.")

counts <- counts[, common_cells, drop = FALSE]
meta <- meta[match(common_cells, rownames(meta)), , drop = FALSE]
min_cells <- as.integer(Sys.getenv("MIN_CELLS_PER_PSEUDOBULK", unset = "30"))

group_key <- paste(meta$condition_std, meta$donor_std, meta$sample_std, meta$celltype_std, meta$region_std, sep = "||")
group_counts <- table(group_key)
keep <- group_key %in% names(group_counts[group_counts >= min_cells])
counts <- counts[, keep, drop = FALSE]
meta <- meta[keep, , drop = FALSE]
group_key <- group_key[keep]
message("Cells retained: ", ncol(counts))
message("Pseudobulk groups retained: ", length(unique(group_key)))

group_fac <- factor(group_key)
design <- sparse.model.matrix(~ 0 + group_fac)
colnames(design) <- levels(group_fac)
pb <- counts %*% design
pb_meta <- data.table(group_key = colnames(pb))
parts <- tstrsplit(pb_meta$group_key, "\\|\\|")
pb_meta[, condition := parts[[1]]]
pb_meta[, donor_id := parts[[2]]]
pb_meta[, sample_id := parts[[3]]]
pb_meta[, cell_type := parts[[4]]]
pb_meta[, brain_region := parts[[5]]]
pb_meta[, dataset_id := "GSE281069"]
pb_meta[, species := "human"]
pb_meta[, pseudobulk_id := paste(dataset_id, condition, donor_id, sample_id, cell_type, brain_region, sep = "::")]
pb_meta[, n_cells := as.integer(group_counts[group_key])]
pb_meta[, matrix_source := rds_path]
colnames(pb) <- pb_meta$pseudobulk_id

expr_dt <- as.data.table(as.matrix(pb), keep.rownames = "gene")
fwrite(expr_dt, file.path(out_dir, "gse281069_pseudobulk_counts.csv.gz"))
fwrite(pb_meta, file.path(out_dir, "gse281069_pseudobulk_metadata.csv"))
message("Wrote Seurat pseudobulk export to ", out_dir)
