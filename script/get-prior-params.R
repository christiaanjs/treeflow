covToPrecision <- function(x){
  1.0/log(x^2 + 1)
}

precisionToCov <- function(x){
  sqrt(exp(1.0/x) - 1)
}

quantiles <- c(0.025, 0.975)
precisionPriorInterval <- function(params){
  qgamma(quantiles, params[1], rate=params[2])
}
paramsToCov <- function(params){
  precisionToCov(rev(precisionPriorInterval(params)))
}

covInterval <- c(0.1, 0.5)
precisionInterval <- rev(precisionToCov(covInterval))
precisionPriorRes <- rriskDistributions::get.gamma.par(p=quantiles, precisionInterval)
precisionPriorRate <- precisionPriorRes["rate"]
precisionPriorShape <- precisionPriorRes["shape"]

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
