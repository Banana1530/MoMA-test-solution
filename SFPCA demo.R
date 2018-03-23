library('Rcpp')
library('RcppArmadillo')
sourceCpp('sfpca.cpp')


#-------------------
# Util function
#-------------------
norm_vec <- function(x) sqrt(sum(x^2))

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

uni <- function(n){
  u_1 <- as.vector(rnorm(n)) 
  return(u_1/norm_vec(u_1))
}

#-------------------
# Generate data
#-------------------
n <- 200
ind <- as.vector(seq(n))
u_1 <- uni(n)
u_2 <- uni(n)
u_3 <- uni(n)
eps <- matrix(rnorm(n*n),n,n)
eps <- eps/20

# Sinusoidal
v_1 <- sin((ind+15)*pi/17);v_1[floor(7/20*n):n]=0;
v_1 <- v_1/norm_vec(v_1);
# Gaussian-modulated sinusoidal
v_2 <- as.vector(exp(-(ind-100)^2/650)*sin((ind-100)*2*pi/21)); 
v_2[0:floor(7/20*n)]=0;v_2[floor(130/200*n):n] = 0;
v_2 <- v_2/norm_vec(v_2);
# Sinusoidal
v_3 <- sin((ind-40)*pi/30);v_3[0:floor(130/200*n)]=0;
v_3 <- v_3/norm_vec(v_3);

plot(v_1,type = 'l',ylim=c(-0.3,0.3));
lines(v_2,col='blue');
lines(v_3,col='red')

X <-  n/4*u_1 %*% t(v_1) +eps + n/5*u_2 %*% t(v_2)
# Noise proportion
print(norm(X) /norm(eps))

#-------------------
# Model setup
#-------------------
P_u <- "l1"
P_v <- "l2"
O_u <- SSD(n)
O_v <- O_u 
# Smoothness
a_u <- 10
a_v <- 10
# Sparsity
lam_u <- 6
lam_v <- 6

#-------------------
# Demo 1, signal recovery
#-------------------
res1 <- sfpca(X,
              O_u,O_v,
              1,1,
              lambda_u=5,lambda_v=5,
              "l1","l1",
              1e-9,1e+5)
plot(res1$v,type='l')  



res2 <- sfpca(res1$DeflatedX,
              O_u,O_v,
              1,1,
              lambda_u=6.5,lambda_v=6.5,
              "l1","l1",
              1e-9,1e+5)

plot(res2$v,type='l')  


#-------------------
# Demo 2, effects of different penalty levels
#-------------------
sm_set = c(0.1,1,10,100)
sp_set = c(1,10,20,30)
par(mfrow=c(length(sm_set),length(sp_set)))
for(sm in sm_set){
  for(sp in sp_set){
    res <- sfpca(X,
                  O_u,O_v,
                  sm,sm,
                  lambda_u=sp,lambda_v=sp,
                  "l1","l1",
                  1e-9,1e+5)
    plot(res$v,type='l')  
  }
}
title(main="Effects of penalty parameters", 
      xlab="Sparsity", ylab="Smoothness")





