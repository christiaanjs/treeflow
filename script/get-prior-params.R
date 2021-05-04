parser <- argparse::ArgumentParser(description="Get relaxed clock prior parameters")
parser$add_argument("alpha", metavar="A", type="double", help="Significance level for quantiles")
parser$add_argument("covInterval", metavar="COV", type="double", nargs=2, help="Coefficient of variation interval")
parser$add_argument("meanInterval", metavar="MEAN", type="double", nargs=2, help="Mean in real space interval")
parser$add_argument("output", metavar="OUTPUT", type="character", help="Output file (yaml)")
args <- parser$parse_args()

covToPrecision <- function(x){
  1.0/log(x^2 + 1)
}

precisionToCov <- function(x){
  sqrt(exp(1.0/x) - 1)
}

alpha <- args$alpha
quantiles <- c(alpha/2.0, 1-alpha/2.0)


precisionPriorInterval <- function(params){
  qgamma(quantiles, params[1], rate=params[2])
}

covInterval <- args$covInterval
precisionInterval <- rev(covToPrecision(covInterval))
precisionPriorRes <- rriskDistributions::get.gamma.par(p=quantiles, precisionInterval, plot=F)
precisionPriorRate <- precisionPriorRes["rate"]
precisionPriorShape <- precisionPriorRes["shape"]

# Check by sampling
# sampleSize <- 1000
# precisionPriorSamples <- rgamma(sampleSize, shape=precisionPriorShape, rate=precisionPriorRate) 
# covPriorSamples <- precisionToCov(precisionPriorSamples)
# covSampleQuantiles <- quantile(covPriorSamples, p=quantiles)
# covSampleQuantiles

lognormalMeanInRealSpace <- function(mu, precision){
  variance <- 1/precision
  exp(mu + variance/2)
}

lognormalMuFromMean <- function(mean, precision){
  # Increasing function of precision
  # Increasing function of mean
  variance <- 1/precision
  log(mean)-variance/2
}

meanInterval <- args$meanInterval
muInterval <- lognormalMuFromMean(meanInterval, precisionInterval)
muPriorRes <- rriskDistributions::get.norm.par(p=quantiles, muInterval, plot=F)
muPriorMean <- muPriorRes[["mean"]]
muPriorSd <- muPriorRes[["sd"]]

# Check by sampling
# muPriorSamples <- rnorm(sampleSize, mean=muPriorMean, sd=muPriorSd)
# sdSamples <- sqrt(1.0 / precisionPriorSamples)
# rateSamples <- lapply(1:1000, function(i) rlnorm(sampleSize, muPriorSamples[i], sdSamples[i]))
# rateMeans <- sapply(rateSamples, mean)
# rateSds <- sapply(rateSamples, sd)
# rateCovs <- rateSds / rateMeans
# quantile(rateMeans, quantiles)
# quantile(rateCovs, quantiles)

outputList <- list(precision=precisionPriorRes, mu=muPriorRes)
yaml::write_yaml(outputList, args$output)
