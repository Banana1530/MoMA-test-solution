#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;
using namespace Rcpp;
using namespace arma;
// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

typedef arma::vec (*PF)(arma::vec, double);


arma::vec lasso (arma::vec x, double l){
  return  sign(x)%max(abs(x) - l,zeros(size(x)));
}

double mat_norm(arma::vec u, arma::mat S_u){
  return sqrt(as_scalar(u.t()*S_u*u));
}


PF prox_op(std::string type){
  if(type.compare("l1") ==0 || type.compare("L1") == 0){
    return lasso;
  }
  else 
    throw std::invalid_argument(type + "is not currently supported");
}


int myassert(bool flag,char * info)

{
  if (flag == 0){
    throw std::invalid_argument(info);
  }
  return 0;
};

// [[Rcpp::export]]
extern "C" SEXP sfpca(arma::mat X,
                    arma::mat Omega_u,
                    arma::mat Omega_v,
                    double alpha_u=0,
                    double alpha_v=0,
                    double lambda_u=0,
                    double lambda_v=0,
                    std::string P_u="NotSupplied",
                    std::string P_v="NotSupplied",
                    double EPS = 1e-4,
                    long MAX_ITER = 1e+3
                        ) {

  
  
  int n = X.n_rows, p = X.n_cols;

  arma::mat U;  arma::vec s;  arma::mat V;  svd(U,s,V,X);
  
  arma::vec u = U.col(0);
  arma::vec v = V.col(0);
  arma::vec oldu = zeros(size(u));
  arma::vec oldv = zeros(size(v));
  arma::vec oldui = zeros(size(u));
  arma::vec oldvi = zeros(size(v));
  
  arma::mat Su; Su.eye(size(Omega_u)); Su += n*alpha_u*Omega_u;
  arma::mat Sv; Sv.eye(size(Omega_v)); Sv += p*alpha_v*Omega_v;

  double Lu = eig_sym(Su).max() + 0.01;
  double Lv = eig_sym(Sv).max() + 0.01;

  PF prox_u = prox_op(P_u);
  PF prox_v = prox_op(P_v);
  


  int iter = 0;
  int indu = 1;
  int indv = 1;
  int indo = 1;
  while(indo>EPS && iter<MAX_ITER){
    oldu = u;
    oldv = v;
    
    indu = 1;
    while(indu>EPS){
      oldui = u;
      u = prox_u(u+(X*v-Su*u)/Lu,lambda_u/Lu);
      if(norm(u)>0){
        u /= mat_norm(u,Su);
      }
      else{
        u.zeros();
      }
      indu = norm(u-oldui)/norm(oldui);
    }
    
    indv = 1;
    while(indv>EPS){
      oldvi = v;
      v = prox_v(v+(X.t()*u-Sv*v)/Lv,lambda_v/Lv);
      if(norm(v)>0){
        v /= mat_norm(v,Sv);
      }
      else{
        v.zeros();
      }
      indv = norm(v-oldvi)/norm(oldvi);
    }
    indo = norm(oldu-u)/norm(oldu) + norm(oldv-v)/norm(oldv);
    iter++;
  }
    
  u = u/norm(u);
  v = v/norm(v);
  double d = as_scalar(u.t() * X * v);
  return Rcpp::List::create(
    Rcpp::Named("u") = u,
    Rcpp::Named("v") = v,
    Rcpp::Named("d") = d,
    Rcpp::Named("DeflatedX") = X - d*u*v.t());

}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be autoarma::matically
// run after the compilation.
//

/*** R

*/
