library(TDA)
library(TDAmapper)
library(ggplot2)
library(igraph)
library(locfit)
library(ks)
library(networkD3)


BOTTLENECKDIST <- 1
BRCDE <- 0

#### open the file
IDICT <- 1
dict <- c("originalSamples.csv", "GPWGAN_generated.csv", "WGAN_generated.csv", "WAE_generated.csv", "VAE_generated.csv")

#### Rips Diagram
if (BOTTLENECKDIST == 1) {
  nSim <- 20
  batch <- 750
  maxBatch <- 3000
  maxscale <- 1 # limit of the filtration
  maxdimension <- 1 # H-dimension
  
  N <- maxBatch / batch
  IDICT <- 1
  flnm <- dict[IDICT]
  xfraudsA <- read.table( paste("/home/jeremy/Documents/SnT/aa_code/app_topwgan/", flnm, sep="") , sep=',')
  lastcolumn <- dim(xfraudsA)[2]# - 1
  
  totBottleneck <- matrix(data=0, nrow=nSim, ncol=4) #c(0, 0, 0, 0)
  for (iSim in seq(1, nSim)) {
    for (idict in seq(1, 4)) {
      flnm <- dict[idict + 1]
      xfraudsB <- read.table( paste("/home/jeremy/Documents/SnT/aa_code/app_topwgan/", flnm, sep="") , sep=',')
      
      n0 <- 1
      for (nbatch in seq(0, maxBatch - batch, batch)) {
        print(paste("iSim: ", iSim, "idict: ", idict, "nbatch: ", n0 + batch - 1))
        print("=============")
        
        # we shuffle the data to remove any bias
        dfA <- xfraudsA[sample(nrow(xfraudsA)),]
        dfB <- xfraudsB[sample(nrow(xfraudsB)),]
        
        Diag1 <- ripsDiag(X = dfA[n0:(n0 + batch - 1),1:lastcolumn], maxdimension, maxscale, library = "GUDHI", printProgress = TRUE)
        Diag2 <- ripsDiag(X = dfB[n0:(n0 + batch - 1),1:lastcolumn], maxdimension, maxscale, library = "GUDHI", printProgress = TRUE)
        
        curBottleneck <- bottleneck(Diag1[["diagram"]], Diag2[["diagram"]])
        print(curBottleneck)
        totBottleneck[iSim, idict] <- totBottleneck[iSim, idict] + curBottleneck
        n0 <- (n0 + batch)
      }
      # we compute the average of the bottleneck distance
      #totBottleneck[idict] <- totBottleneck[idict] / N
      totBottleneck[iSim, idict] <- totBottleneck[iSim, idict] / N
    }
  }
  print(totBottleneck)
}

#### Barcode
PERSDIAG <- 0
if (BRCDE == 1) {
  n0 <- 1
  batch <- 250
  maxBatch <- 2500
  maxscale <- 1 # limit of the filtration
  maxdimension <- 1 # H-dimension
  
  N <- maxBatch / batch
  IDICT <- 4
  flnm <- dict[IDICT]
  print(flnm)
  xfraudsA <- read.table( paste("/home/jeremy/Documents/SnT/aa_code/app_topwgan/", flnm, sep="") , sep=',')
  lastcolumn <- dim(xfraudsA)[2]# - 1
  
  dfA <- xfraudsA[sample(nrow(xfraudsA)),]
  Diag1 <- ripsDiag(X = dfA[n0:(n0 + batch - 1),1:lastcolumn], maxdimension, maxscale, library = "GUDHI", printProgress = TRUE)
  if (PERSDIAG == 0) {
    plot(Diag1[["diagram"]], barcode = TRUE)
  } else {
    plot(Diag1[["diagram"]], rotated=TRUE, barcode = FALSE)
  }
}