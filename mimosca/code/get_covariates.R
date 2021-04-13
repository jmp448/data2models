library(tidyverse)
library(Matrix)
library(elasticnet)

# get counts per cell
ncount <- tibble("cell"=colnames(day0), "reads"=unname(colSums(day0))) %>%
  write_tsv("../data/day0/covariates.tsv")

# one-hot encode guides 
guides = read_tsv("../data/day0/guide_cell_map_singlets.tsv") %>%
  select(-tf) %>%
  mutate(val=1) %>%
  pivot_wider(names_from=grna, values_from=val, values_fill=0)

ncount = filter(ncount, cell %in% guides$cell)
guides = filter(guides, cell %in% ncount$cell)

design = inner_join(ncount, guides, by="cell") %>%
  write_tsv("../data/day0/design_guide_counts.tsv")

# load expression data
day0 <- readRDS("../data/day0/full_data.rds")[,design$cell] %>%
  t %>%
  as_tibble(rownames="cell") %>%
  write_tsv("../data/day0/exp_singlets.tsv")

