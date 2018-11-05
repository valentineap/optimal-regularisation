# optimal-regularisation

This project provides a reference implementation of the algorithms described in:

  Valentine, A.P. and Sambridge, M., [Optimal regularisation for a class of linear inverse problem](http://dx.doi.org/10.1093/gji/ggy303), *Geophysical Journal International*, 215(2), pp.1003--1021, 2018.

Specifically, these algorithms are intended to solve linear regression problems of the form

minimize_m ||Gm - d||^2 +||D(eps) m||^2

where D(eps) is a regularisation operator that depends on one or more tuneable parameters, eps. We use a hierarchical Bayesian approach to select the 'optimal' choice of eps. In addition to a general-purpose algorithm, we provide one that is substantially more efficient for the common case where the regularisation operator is of Tikhonov form. For full details, please refer to the paper.
