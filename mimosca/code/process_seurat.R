library(Seurat)

bmdc <- readRDS("../data/obs/full_data.srt")

bmdc <- bmdc %>%
  NormalizeData %>%
  FindVariableFeatures(selection.method = "vst", nfeatures = 1000) %>%
  ScaleData %>%
  RunPCA

bmdc <- bmdc %>%
  FindNeighbors(dims=1:15) %>%
  FindClusters(res=0.5) %>%
  RunUMAP(dims=1:15)

DimPlot(bmdc, group.by="seurat_clusters")