// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// rcpparma_hello_world
arma::mat rcpparma_hello_world();
RcppExport SEXP _sfpca_rcpparma_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpparma_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_outerproduct
arma::mat rcpparma_outerproduct(const arma::colvec& x);
RcppExport SEXP _sfpca_rcpparma_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_innerproduct
double rcpparma_innerproduct(const arma::colvec& x);
RcppExport SEXP _sfpca_rcpparma_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_bothproducts
Rcpp::List rcpparma_bothproducts(const arma::colvec& x);
RcppExport SEXP _sfpca_rcpparma_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}
// sfpca
extern "C" SEXP sfpca(arma::mat X, arma::mat Omega_u, arma::mat Omega_v, double alpha_u, double alpha_v, double lambda_u, double lambda_v, std::string P_u, std::string P_v, double EPS, long MAX_ITER);
RcppExport SEXP _sfpca_sfpca(SEXP XSEXP, SEXP Omega_uSEXP, SEXP Omega_vSEXP, SEXP alpha_uSEXP, SEXP alpha_vSEXP, SEXP lambda_uSEXP, SEXP lambda_vSEXP, SEXP P_uSEXP, SEXP P_vSEXP, SEXP EPSSEXP, SEXP MAX_ITERSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Omega_u(Omega_uSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Omega_v(Omega_vSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_u(alpha_uSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_v(alpha_vSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_u(lambda_uSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_v(lambda_vSEXP);
    Rcpp::traits::input_parameter< std::string >::type P_u(P_uSEXP);
    Rcpp::traits::input_parameter< std::string >::type P_v(P_vSEXP);
    Rcpp::traits::input_parameter< double >::type EPS(EPSSEXP);
    Rcpp::traits::input_parameter< long >::type MAX_ITER(MAX_ITERSEXP);
    rcpp_result_gen = Rcpp::wrap(sfpca(X, Omega_u, Omega_v, alpha_u, alpha_v, lambda_u, lambda_v, P_u, P_v, EPS, MAX_ITER));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_sfpca_rcpparma_hello_world", (DL_FUNC) &_sfpca_rcpparma_hello_world, 0},
    {"_sfpca_rcpparma_outerproduct", (DL_FUNC) &_sfpca_rcpparma_outerproduct, 1},
    {"_sfpca_rcpparma_innerproduct", (DL_FUNC) &_sfpca_rcpparma_innerproduct, 1},
    {"_sfpca_rcpparma_bothproducts", (DL_FUNC) &_sfpca_rcpparma_bothproducts, 1},
    {"_sfpca_sfpca", (DL_FUNC) &_sfpca_sfpca, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_sfpca(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
