library(tidyverse)
library(vroom)
library(Seurat)
library(Matrix)

# observational data
gene_names <- vroom("../data/obs/GSE90063_dc0hr_umi_wt.txt", skip=1, col_select=1, col_names="gene") %>%
  mutate(g=str_split(gene, "_")) %>%
  rowwise %>%
  mutate(ensg=g[2]) %>%
  mutate(hgnc=g[4], .keep="unused") %>%
  ungroup
exp <- vroom("../data/obs/GSE90063_dc0hr_umi_wt.txt") %>%
  select(-c(1)) %>%
  as.matrix %>%
  `rownames<-`(gene_names$hgnc)
sc <- CreateSeuratObject(counts=as.matrix(exp))
saveRDS(sc, "../data/obs/full_data.srt")

# day 0 perturbed data
exp <- vroom("../data/day0/exp.txt", skip=3, col_names=c("gene_id", "cell_id", "counts"))
cell_names <- vroom("../data/day0/cellnames.csv", col_names=c("cell_id", "cell"), skip=1)
gene_names <- vroom("../data/day0/genenames.csv", col_names=c("gene_id", "gene_names"), skip=1) %>%
  mutate(hgnc=str_extract(gene_names, "[^_]+$")) %>%
  mutate(ensg=str_extract(gene_names, "[^_]+")) #note - 11 duplicated hgnc, 0 duplicated ensg
dup.genes <- gene_names %>%
  group_by(hgnc) %>%
  count %>%
  filter(n>1) %>%
  .$hgnc

cbc_gbc <- read_csv("../data/day0/cbc_gbc_dict.csv", col_names=c("grna", "cell")) %>%
  mutate(cell=str_split(cell, ",")) %>%
  rowwise() %>%
  mutate(tf=str_split(grna, "_")) %>%
  mutate(tf=tf[2]) %>%
  ungroup() %>%
  unnest(cell) %>%
  write_tsv("../data/day0/guide_cell_map.tsv")

singlets <- cbc_gbc %>%
  group_by(cell) %>%
  count %>%
  filter(n==1) %>%
  .$cell

cbc_gbc_singlets <- cbc_gbc %>%
  filter(cell %in% singlets) %>%
  write_tsv("../data/day0/guide_cell_map_singlets.tsv")

day0 <- sparseMatrix(i=exp$gene_id, j=exp$cell_id, x=exp$counts, dimnames=list(gene_names$hgnc, cell_names$cell))
day0 <- day0[setdiff(gene_names$hgnc, dup.genes),] # remove ambiguous hgnc

saveRDS(obj = day0, file="../data/day0/full_data.rds")
