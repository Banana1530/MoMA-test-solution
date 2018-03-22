context("SFPCA testing")

SSD <- function(n){
  a <- 6*diag(n)
  for(i in 1:n){
    for(j in 1:n){
      if(abs(i-j) == 1) a[i,j] = -3;
      if(abs(i-j) == 2) a[i,j] = 1;
    }
  }      
  return(a);
}
test_that("svd checking", {
    for(i in 1:10){
    n <- 20
    X <- matrix(rnorm(n*n),n,n)
    O_u <- SSD(n)
    O_v <- SSD(n)
    res_svd <- svd(X)
    res_sfpca <- sfpca(X, O_u,O_v,
                               0,0,
                               lambda_u=0,lambda_v=0,
                               "l1","l1",
                               EPS=1e-9,MAX_ITER=1e+5)
    expect_equal(norm(res_svd$v[,1]-res_sfpca$v),0,tolerance=1e-9)
  }
})

