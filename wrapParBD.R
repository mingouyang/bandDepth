ompBD2 <- function(x)
{
  if(!is.loaded("ompBD")) {
    dyn.load("ompBD.so")
  }
  d <- dim(x)
  rst <- .C("ompBD2",
  as.integer(d[1]),
  as.integer(d[2]),
  x=as.double(unlist(x)),
  depth=double(length=d[1]))
  rst <- rst[["depth"]]
  return(rst)
}

ompBD3 <- function(x)
{
  if(!is.loaded("ompBD")) {
    dyn.load("ompBD.so")
  }
  d <- dim(x)
  rst <- .C("ompBD3",
  as.integer(d[1]),
  as.integer(d[2]),
  x=as.double(unlist(x)),
  depth=double(length=d[1]))
  rst <- rst[["depth"]]
  return(rst)
}

askewBD2 <- function(x)
{
  if(!is.loaded("ompBD")) {
    dyn.load("ompBD.so")
  }
  d <- dim(x)
  rst <- .C("askewBD2",
  as.integer(d[1]),
  as.integer(d[2]),
  x=as.double(unlist(x)),
  depth=double(length=d[1]))
  rst <- rst[["depth"]]
  return(rst)
}

lowDimBD2 <- function(x)
{
  if(!is.loaded("ompBD")) {
    dyn.load("ompBD.so")
  }
  d <- dim(x)
  rst <- .C("lowDimBD2",
  as.integer(d[1]),
  as.integer(d[2]),
  x=as.double(unlist(x)),
  depth=double(length=d[1]))
  rst <- rst[["depth"]]
  return(rst)
}

gpuBD2 <- function(x)
{
  if(!is.loaded("gpuBD")) {
    dyn.load("gpuBD.so")
  }
  d <- dim(x)
  rst <- .C("gpuBD2",
  as.integer(d[1]),
  as.integer(d[2]),
  x=as.double(unlist(x)),
  depth=double(length=d[1]))
  rst <- rst[["depth"]]
  return(rst)
}

multigpuBD2 <- function(x)
{
  if(!is.loaded("gpuBD")) {
    dyn.load("gpuBD.so")
  }
  d <- dim(x)
  rst <- .C("multigpuBD2",
  as.integer(d[1]),
  as.integer(d[2]),
  x=as.double(unlist(x)),
  depth=double(length=d[1]))
  rst <- rst[["depth"]]
  return(rst)
}

gpuBD3 <- function(x)
{
  if(!is.loaded("gpuBD")) {
    dyn.load("gpuBD.so")
  }
  d <- dim(x)
  rst <- .C("gpuBD3",
  as.integer(d[1]),
  as.integer(d[2]),
  x=as.double(unlist(x)),
  depth=double(length=d[1]))
  rst <- rst[["depth"]]
  return(rst)
}

multigpuBD3 <- function(x)
{
  if(!is.loaded("gpuBD")) {
    dyn.load("gpuBD.so")
  }
  d <- dim(x)
  rst <- .C("multigpuBD3",
  as.integer(d[1]),
  as.integer(d[2]),
  x=as.double(unlist(x)),
  depth=double(length=d[1]))
  rst <- rst[["depth"]]
  return(rst)
}
