library(tidyverse)
library(elasticnet)


covariates =read_tsv("../data/day0/covariates.tsv") %>%
  filter(cell %in% design$cell)


x = inner_join(design, covariates, by="cell")

exp = t(readRDS("../data/day0/full_data.rds")[,x$cell])

x = select(x, -cell) %>% as.matrix()

m <- enet(x, exp[,1], lambda=0.5, max.steps = 10000)






