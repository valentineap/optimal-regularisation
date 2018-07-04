'''
v1.1 -- June 2018

This code provides a reference implementation of the algorithms described in:
   Valentine, A.P. & Sambridge, M., `Optimal regularisation for a class of linear
      inverse problem', Geophys. J. Int., 2018.

All code is provided 'as is' and without warranty. We welcome bug-fixes, comments and
suggestions, but we are unable to provide users with detailed technical support. A Git
repository containing this code can be found online at:
https://github.com/valentineap/optimal-regularisation
This will be used to distribute any future versions of this code.

This file provides the following routines:
        ** General covariance matrices **
    getLogProbGeneral(GTG,GTd,iCm,k=0)
    getLogProbDerivGeneral(GTG,GTd,iCm,Cm=None)
    findOptimalXi(GGTG<GTd,xi0,xi2cov,hprior=None,bounds=None,report=None)
    solveLSQgeneral(GTG,GTd,xi0,xi2cov,hprior=None,bounds=None,report=None,fullOutput=False)

        ** Tikhonov-style regularisation **
    getLogProbTik(GTG,GTd,H,alpha,beta,k=0)
    getAlphaBetaDerivatives(GTG,GTd,H,alpha,beta)
    getAlphaBetaSecondDerivatives(GTG,GTd,H, alpha,beta)
    getOptimalAlpha(lam, S, GTd, omega, T, beta,dlhprior=None,alphaMin=1e-12,alphaMax=None)
    getOptimalBeta(gamma,U,GTd,omega,T,alpha,dlhprior=None,betaMin=1e-12,betaMax=None,Uinv=None)
    solveLSQtikhonov(GTG,GTd,alpha = None,beta = None, H = None,hprior=None,fullOutput=False
                        ,alphaMin=1e-12,betaMin=1e-12,alphaMax=None,betaMax=None,guess=None)
The two 'solveLSQ*' routines are likely to be sufficient for many users' purposes.

The interface for each function is documented within its docstring. For full explanation,
please refer to the accompanying paper, and cite it if you make use of these algorithms
within your own work. Variable names within this code are generally chosen to accord with
the notation used in the paper.

To assist the user in understanding how these routines may be employed in practice,
this file also contains the code necessary to generate many of the figures presented
within the paper. To run this code, execute the file directly, e.g.
 > python regularised_lsq.py
Otherwise, this file is intended to be treated as a module and imported into other
scipts (e.g. 'import regularised_lsq as rlsq').


Andrew P. Valentine
Research School of Earth Sciences
The Australian National University
andrew.valentine@anu.edu.au

'''

import numpy as np
import scipy.optimize as optim

# The following routines are intended to work for general covariance matrices
# as described in Section 3.1 of the paper.

# Eq. (18)
def getLogProbGeneral(GTG,GTd,iCm,k=0):
    ''' Compute log(P[ d' | Cm^{-1} ]) for a general covariance matrix.
    Inputs:
    GTG, GTd -- the relevant linear system
    iCm -- the *inverse* covariance matrix, Cm^{-1}
    k -- Normalisation constant (parenthesised terms in V&S)
    '''
    ldiCm = np.linalg.slogdet(iCm)[1]
    Q = GTG+iCm
    ldQ = np.linalg.slogdet(Q)[1]
    return 0.5*(k+ldiCm-ldQ+GTd.dot(np.linalg.inv(Q).dot(GTd)))

# Eq. (19)
def getLogProbDerivGeneral(GTG,GTd,iCm,Cm=None):
    ''' Compute matrix representing derivative of log(P[ d' | Cm^{-1} ]) with
    respect to each of the elements of Cm^{-1}.
    Inputs:
    GTG,GTd -- the relevant linear system
    iCm -- the inverse covariance matrix, Cm^{-1}
    Cm -- the 'forward' covariance matrix. Computed by inverting iCm if not supplied.
    '''
    if Cm is None:Cm=np.linalg.inv(iCm)
    iQ = np.linalg.inv(GTG+iCm)
    GTdp = GTd.reshape([GTd.shape[0],1])
    GTddTG = GTdp.dot(GTdp.T)
    return 0.5*(Cm - iQ - iQ.dot(GTddTG.dot(iQ)))

# Find zero of eq. (21)
def findOptimalXi(GTG,GTd,xi0,xi2cov,hprior=None,bounds=None,report=None):
    ''' Given a parameterised model covariance matrix Cm = Cm(xi), find the optimal
    value of xi to use within a given inverse problem.

    Inputs:
    GTG,GTd -- the relevant linear system. Note that GTd here is GTd' in the paper,
               i.e. G^T(d - Gm_p)
    xi0     -- Initial guess for parameter vector xi
    xi2cov  -- Callable specifiying connection between xi and covariance matrices.
               Should return a tuple, (Cm, iCm, deriv_iCm) = xi2Cm(xi), where
               Cm        -- Model covariance matrix, Cm(xi)
               iCm       -- Inverse of model covariance matrix, [Cm(xi)]^{-1}
               deriv_iCm -- Three-dimensional array containing derivatives of the
                            inverse covariance matrix w.r.t. the parameters xi,
                            such that deriv_iCm[:,:,i] = \partial (Cm)/\partial xi_i
    hprior  -- Callable defining hyper-prior on xi. Should return a tuple,
               (lP,dlP) = hprior(xi), where
               lP  -- Log of prior probability associated with hyperparameters xi
               dlP -- Vector representing the derivative of Log[P(xi)] with respect to
                      each component, evaluated at the point xi, i.e.
                      {d[log P]/d(xi_i)|_xi , i=1..K}
               If None, a uniform hyperprior is assumed.
    bounds  -- List of K tuples (lo, up) providing lower- and upper-bounds on the
               permissible values of each component of xi. lo and/or up may be 'None'
               to signify no bound.
    report  -- Callable report(xi), passed to l_bfgs optimizer and called at each
               iteration of the optimization procedure. May be used to print progress
               information. May be 'None'.
    Returns (xi_opt, info) where
    xi_opt -- Optimal value of xi
    info   -- Dictionary containing output from l_bfgs optimizer.
    '''
    K = xi0.shape[0]
    M = GTd.shape[0]
    GTddTG = GTd.reshape([M,1]).dot(GTd.reshape([1,M]))
    def objective(xi):
        Cm,iCm,drv = xi2cov(xi)
        if Cm is None:
            if iCm is None: raise ValueError, "xi2cov does not appear to return values"
            Cm = np.linalg.inv(iCm)
        if iCm is None:
            iCm = np.linalg.inv(Cm)
        ldiCm = np.linalg.slogdet(iCm)[1]
        Q = GTG+iCm
        iQ = np.linalg.inv(Q)
        ldQ = np.linalg.slogdet(Q)[1]
        logProb =  0.5*(ldiCm-ldQ+GTd.dot(iQ.dot(GTd)))
        dlPdC = 0.5*(Cm - iQ - iQ.dot(GTddTG.dot(iQ)))
        dlPdXi = np.zeros(K)
        for i in range(K):
            dlPdXi[i] = np.trace(drv[:,:,i].T.dot(dlPdC))
        if hprior is None:
            lPXi,dlPXi=(0.,0.)
        else:
            lPXi,dlPXi = hprior(xi)
        return -logProb-lPXi,-dlPdXi-dlPXi
    xi_opt,f_xi_opt,info = optim.fmin_l_bfgs_b(objective,xi0,callback=report,disp=1,bounds=bounds)
    return xi_opt,info


# Wrap findOptimalXi into routine that solves inverse problem
def solveLSQgeneral(GTG,GTd,xi0,xi2cov,hprior=None,bounds=None,report=None,fullOutput=False):
    '''
    Solve a least-squares problem for a general choice of covariance matrix.

    Inputs:
    GTG,GTd    -- the relevant linear system. Note that GTd here is GTd' in the paper,
                  i.e. G^T(d - Gm_p)
    xi0        -- Initial guess for parameter vector xi
    xi2cov     -- Callable specifiying connection between xi and covariance matrices.
                  Should return a tuple, (Cm, iCm, deriv_iCm) = xi2Cm(xi), where
                  Cm        -- Model covariance matrix, Cm(xi)
                  iCm       -- Inverse of model covariance matrix, [Cm(xi)]^{-1}
                  deriv_iCm -- Three-dimensional array containing derivatives of the
                               inverse covariance matrix w.r.t. the parameters xi,
                               such that deriv_iCm[:,:,i] = \partial (Cm)/\partial xi_i
    hprior     -- Callable defining hyper-prior on xi. Should return a tuple,
                  (lP,dlP) = hprior(xi), where
                  lP  -- Log of prior probability associated with hyperparameters xi
                  dlP -- Vector representing the derivative of Log[P(xi)] with respect to
                         each component, evaluated at the point xi, i.e.
                         {d[log P]/d(xi_i)|_xi , i=1..K}
                  If None, a uniform hyperprior is assumed.
    bounds     -- List of K tuples (lo, up) providing lower- and upper-bounds on the
                  permissible values of each component of xi. lo and/or up may be 'None'
                  to signify no bound.
    report     -- Callable report(xi), passed to l_bfgs optimizer and called at each
                  iteration of the optimization procedure. May be used to print progress
                  information. May be 'None'.
    fullOutput -- Set to 'True' to return additional information (see below)

    If fullOutput = False, returns m where:
    m          -- Vector of best-fitting model coefficients.
    If fullOutput = True, returns (m, pCov, xiOpt) where:
    m          -- Vector of best-fitting model coefficients
    pCov       -- Posterior covariance matrix
    xiOpt      -- Optimal hyper-parameter vector

    '''
    xi_opt,info = findOptimalXi(GTG,GTd,xi0,xi2cov,hprior,bounds,report)
    Cm,iCm,drv = xi2cov(xi_opt)
    pCov = np.linalg.inv(GTG+iCm)
    if fullOutput:
        return pCov.dot(GTd), pCov, xi_opt
    else:
        return pCov.dot(GTd)



# The following routines are for the special case of 'Tikhonov Regularisation' as
# discussed in Section 3.2 of the paper.

# Evaluate eq. (18) in the special case of Tikhonov
def getLogProbTik(GTG,GTd,H,alpha,beta,k=0):
    ''' Evaluate log[P(d' | a,b )] for Tikhonov regularisation, where
    C_m^{-1} = a^2 I + b^2 H.
    Inputs:
    GTG, GTd -- The linear system
    H        -- Smoothing matrix. May be None if not required.
    alpha    -- Weight for norm-damping term
    beta     -- Weight for smoothing term (ignored if H = None)
    k        -- Optional; normalisation constant (parenthesised term in V&S)
    '''
    M = GTG.shape[0]
    iCm = alpha**2 *np.eye(M)
    if H is not None:
        iCm+= beta**2 * H
    return getLogProbGeneral(GTG,GTd,iCm,k)

# Eq. 23
def getAlphaBetaDerivatives(GTG,GTd,H,alpha,beta):
    '''
    Evaluate:
        \partial \log P[d' | alpha, beta] / \partial alpha
        \partial \log P[d' | alpha, beta] / \partial beta
    at a given point.
    Inputs:
    GTG, GTd -- The linear system
    H        -- The smoothing matrix. May be None if not required. Can also give
                in the form of a tuple (omega,T) such that H = T.dot(np.diag(omega)).dot(T.T)
                i.e. an eigendecomposition of H. If available, this avoids the need to perform
                matrix inversion to obtain both forward and inverse covariance matrices.
    alpha    -- Weight for norm-damping term
    beta     -- Weight for smoothing term (ignored if H = None)
    '''
    M = GTG.shape[0]
    if H is None:
        iCm = alpha**2 *np.eye(M)
        Cm = np.eye(M)/(alpha**2)
    elif type(H) is type(()):
        omega,T = H
        iCm = T.dot(np.diag(alpha**2+beta**2*omega)).dot(T)
        Cm = T.dot(np.diag(1./(alpha**2+beta**2*omega))).dot(T)
    else:
        iCm = alpha**2 * np.eye(M) + beta**2*H
        Cm = np.linalg.inv(iCm)
    Q = np.linalg.inv(GTG+iCm)
    da = alpha*(np.trace(Cm - Q) - GTd.dot(Q.dot(Q).dot(GTd)))
    if H is None:
        db = 0.
    else:
        db = beta * (np.trace(H.dot(Cm - Q)) - GTd.dot(Q.dot(H).dot(Q).dot(GTd)))
    return np.array([da,db])

# Eq. A1
def getAlphaBetaSecondDerivatives(GTG,GTd,H, alpha,beta):
    '''
    Evaluate:
        \partial^2 \log P[d' | alpha, beta] / \partial alpha^2
        \partial^2 \log P[d' | alpha, beta] / \partial beta^2
        \partial^2 \log P[d' | alpha, beta] / \partial alpha \partial beta
    at a given point.
    Inputs:
    GTG, GTd -- The linear system
    H        -- The smoothing matrix. May be None if not required. Can also give
                in the form of a tuple (omega,T) such that H = T.dot(np.diag(omega)).dot(T.T)
                i.e. an eigendecomposition of H. If available, this avoids the need to perform
                matrix inversion to obtain both forward and inverse covariance matrices.
    alpha    -- Weight for norm-damping term
    beta     -- Weight for smoothing term (ignored if H = None)
    '''
    M = GTG.shape[0]
    if H is None:
        iCm = alpha**2 *np.eye(M)
        Cm = np.eye(M)/(alpha**2)
    elif type(H) is type(()):
        omega,T = H
        iCm = T.dot(np.diag(alpha**2+beta**2*omega)).dot(T)
        Cm = T.dot(np.diag(1./(alpha**2+beta**2*omega))).dot(T)
    else:
        iCm = alpha**2 * np.eye(M) + beta**2*H
        Cm = np.linalg.inv(iCm)
    Q = np.linalg.inv(GTG+iCm)

    da_over_alpha =  (np.trace(Cm-Q) - GTd.dot(Q.dot(Q).dot(GTd)))
    d2a = da_over_alpha - 2*alpha**2 * np.trace(Cm.dot(Cm) - Q.dot(Q)) \
                    + 4*alpha**2 *GTd.dot(Q.dot(Q).dot(Q).dot(GTd))

    if H is None:
        d2b = 0.
        d2ab = 0.
    else:
        db_over_beta = (np.trace(H.dot(Cm-Q)) - GTd.dot(Q.dot(H).dot(Q).dot(GTd)))
        d2b = db_over_beta - 2*beta**2 *np.trace(H.dot(Cm.dot(H).dot(Cm) - Q.dot(H).dot(Q))) \
                    +4*beta**2 * GTd.dot(Q.dot(H).dot(Q).dot(H).dot(Q).dot(GTd))
        d2ab = -2*alpha*beta * (np.trace(Cm.dot(H).dot(Cm) - Q.dot(H).dot(Q)) \
                    + GTd.dot(Q.dot(Q.dot(H)+H.dot(Q)).dot(Q).dot(GTd)))
    return d2a,d2b,d2ab

# Eq. 26
def getOptimalAlpha(lam, S, GTd, omega, T, beta,dlhprior=None,alphaMin=1e-12,alphaMax=None):
    '''
    Solve the root-finding problem to determine an optimal choice
    of alpha.

    Inputs:
    lam, S = np.linalg.eigh(GTG + beta**2 H)   }   These quantities
    omega,T = np.linalg.eigh(H)                }   define the linear
    beta                                       }   system to be solved
    dlhprior -- Callable, dlhprior(a), returning
                d{ log P[ d | alpha ]}/d alpha|_{a}
                i.e. the derivative of log hyperprior evaluated at a
                given point.

    alphaMin -- Lower limit for the search. In principle this should
                be 0, but this can cause difficulties in case where
                beta=0. Default value should suffice for most practical
                circumstances.
    alphaMax -- Upper limit for the search. If no value provided, we use
                alphaMax = 100 * lam.max().
    '''
    if alphaMax is None: alphaMax = 100. * lam.max()
    STGTd2 = (S.T.dot(GTd))**2
    if dlhprior is None:
        rootfunc = lambda alpha: (1./(alpha**2 + beta**2*omega) - 1./(alpha**2 + lam) \
                                - STGTd2/((alpha**2+lam)**2)).sum()*alpha
    else:
        rootfunc = lambda alpha: (1./(alpha**2 + beta**2*omega) - 1./(alpha**2 + lam) \
                                - STGTd2/((alpha**2+lam)**2)).sum()*alpha + dlhprior(alpha)
    try:
        alpha0 = optim.brentq(rootfunc,alphaMin,alphaMax)
    except ValueError as err:
        print "Error: Unable to find an optimal value of alpha\n  "\
                        "Search range: [%f,%f]\n  beta: %f\n"\
                        "Raising error from scipy.optimize.brentq..."%(alphaMin,alphaMax,beta)
        raise err
    return alpha0

# Eq. 28
def getOptimalBeta(gamma,U,GTd,omega,T,alpha,dlhprior=None,betaMin=1e-12,betaMax=None,Uinv=None):
    '''
    Solve the root-finding problem to determine an optimal choice
    of beta.

    Inputs:
    gamma, U = np.linalg.eig(np.linalg.inv(H).dot(GTG + alpha**2 I }   These quantities
    omega,T = np.linalg.eigh(H)                                    }   define the linear
    alpha                                                          }   system to be solved

    dlhprior -- Callable, dlhprior(b), returning
                d{ log P[ d | beta ]}/d beta|_{b}
                i.e. the derivative of log hyperprior evaluated at a
                given point.
    betaMin  -- Lower limit for the search. In principle this should
                be 0, but this can cause difficulties in case where
                alpha=0. Default value should suffice for most practical
                circumstances.
    betaMax  -- Upper limit for the search. If no value provided, we use
                betaMax = 100 * gamma.max().
    Uinv     -- np.linalg.inv(U); computed if not provided.
    '''
    if Uinv is None: Uinv = np.linalg.inv(U)
    if betaMax is None: betaMax = 100.* gamma.max()
    denom = U.T.dot(GTd) * Uinv.dot(T.dot(np.diag(1./omega)).dot(T.T)).dot(GTd)
    if dlhprior is None:
        rootfunc = lambda beta: (omega/(alpha**2 + beta**2*omega) - 1./(gamma+beta**2) \
                                - denom/((gamma+beta**2)**2)).sum()
    else:
        rootfunc = lambda beta: (omega/(alpha**2 + beta**2*omega) - 1./(gamma+beta**2) \
                                - denom/((gamma+beta**2)**2)).sum() + dlhprior(beta)
    try:
        beta0 = optim.brentq(rootfunc,betaMin,betaMax)
    except ValueError as err:
        print "Error: Unable to find an optimal value of beta\n  "\
                        "Search range: [%f,%f]\n  alpha: %f\n"\
                        "Raising error from scipy.optimize.brentq..."%(betaMin,betaMax,alpha)
        raise err
    return beta0


# This routine could also be implemented as an iterative scheme, alternating between
# calling getOptimalAlpha() and getOptimalBeta().
def solveLSQtikhonov(GTG,GTd,alpha = None,beta = None, H = None,hprior=None,fullOutput=False \
                    ,alphaMin=1e-12,betaMin=1e-12,alphaMax=None,betaMax=None,guess=None):
    '''
    Solve least-squares problem for a Tikhonov regularisation framework.
    Inputs:
    GTG, GTd     -- The linear system
    H            -- Smoothing matrix. May be None if not required.
    alpha        -- Weight for norm-damping term
    beta         -- Weight for smoothing term (ignored if H = None)
    hprior       -- Callable defining hyperprior on alpha and beta. Should return a tuple,
                    (lp, dlp) where
                    lp  -- log( P[ d' | a,b])
                    dlp -- Array, [d(log P)/da, d(log P)/db ]
    fullOutput   -- Set to True to return additional information
    alphaMin,alphaMax,betaMin,betaMax -- Allowable parameter ranges, if required
    guess        -- Initial guess for optimal (alpha,beta). If None, we first estimate alpha and beta separately.

    If fullOutput=False, returns m where
    m            -- Vector of best-fitting model coefficients
    If fullOutput=True, returns (m, pCov, alpha0,beta0) where
    m            -- Vector of best-fitting model coefficients
    pCov         -- posterior covariance matrix
    alpha0,beta0 -- Optimal regularisation parameters
    '''

    M = GTG.shape[0]
    if H is None: beta = 0.
    if alpha is not None and beta is not None:
        # We have been given values for both alpha and beta; no need to
        # determine either.
        iCm = alpha**2 * np.eye(M)
        if H is not None:
            iCm += beta**2 * H
        pCov = np.linalg.inv(GTG+iCm)
        alpha0=alpha
        beta0=beta
    elif alpha is None and beta is not None:
        # Only need to find alpha
        if H is None:
            lam, S = np.linalg.eigh(GTG)
            # Make some dummy entries for omega, T
            omega = np.zeros(M)
            T = np.eye(M)
        else:
            lam, S = np.linalg.eigh(GTG+beta**2*H)
            omega, T = np.linalg.eigh(H)
        if hprior is None:
            dlhp = None
        else:
            dlhp = lambda a: hprior(np.array([a,beta]))[1][0]
        alpha0 = getOptimalAlpha(lam,S,GTd,omega,T,beta,dlhp)
        beta0 = beta
        pCov = S.dot(np.diag(1./(lam+alpha0**2))).dot(S.T)
    elif alpha is not None and beta is None:
        # Only need to find beta
        omega,T = np.linalg.eigh(H)
        invH = T.dot(np.diag(1./omega)).dot(T.T)
        gamma, U = np.linalg.eig(invH.dot(GTG + alpha**2 * np.eye(M)))
        invU = np.linalg.inv(U)
        if hprior is None:
            dlhp = None
        else:
            dlhp = lambda b: hprior(np.array([alpha,b]))[1][1]
        alpha0 = alpha
        beta0 = getOptimalBeta(gamma,U,GTd,omega,T,alpha,dlhp,Uinv=invU)
        pCov = U.dot(np.diag(1./(gamma+beta0**2))).dot(invU).dot(invH)
    else:
        # Need to find both alpha and beta. One strategy is to alternate
        # the above two methods. However, may be simplest to just use the
        # 'full' theory.
        omega,T = np.linalg.eigh(H)
        def xi2cov(xi):
            # xi = np.array([alpha,beta])
            iCm = xi[0]**2 *np.eye(M) + xi[1]**2*H
            Cm = T.dot(np.diag(1./(xi[0]**2 + xi[1]**2*omega))).dot(T.T)
            deriv_iCm = np.zeros([M,M,2])
            deriv_iCm[:,:,0] = 2.*xi[0]*np.eye(M)
            deriv_iCm[:,:,1] = 2.*xi[1]*H
            return (Cm, iCm, deriv_iCm)
        if guess is None:
            # Estimate alpha for beta=0 and beta for alpha=0 and use these as an initial guess
            lam, S = np.linalg.eigh(GTG)
            if hprior is None:
                dlhpa = None
                dlhpb = None
            else:
                dlhpa = lambda a: hprior(np.array([a,beta]))[1][0]
                dlhpb = lambda b: hprior(np.array([alpha,b]))[1][1]
            gamma, U = np.linalg.eig((T.dot(np.diag(1./omega)).dot(T.T)).dot(GTG))
            xiStart = np.array([getOptimalAlpha(lam,S,GTd,omega,T,0.,dlhpa),getOptimalBeta(gamma,U,GTd,omega,T,0.,dlhpb)])
        else:
            xiStart = np.array(guess)
        xi0,info = findOptimalXi(GTG, GTd, xiStart,xi2cov,hprior,[(alphaMin,alphaMax),(betaMin,betaMax)])
        alpha0,beta0=xi0
        pCov = np.linalg.inv(GTG+alpha0**2*np.eye(M) + beta0**2*H)
    m = pCov.dot(GTd)
    if fullOutput:
        return m, pCov, alpha0,beta0
    else:
        return m



if __name__ == '__main__':
    ''' Generate the figures from the paper, to serve as an example
    for using the above functions.
    '''
    import matplotlib.pyplot as plt
    import scipy.special

    saveFigures=True # Write out figures as .pdf files?
    # def gamma(x,shape,scale):
    #     return x**(shape-1) * np.exp(-x/scale)/(scipy.special.gamma(shape)*scale**shape)
    plt.close('all')
    # (Optional) -- Seed random number generator to allow repeatability
    np.random.seed(42)
    # A simple forward model -- a polynomial
    def poly(x,coeffs):
        y = 0.
        for i in range(coeffs.shape[0]):
            y+=coeffs[i]*x**i
        return y
    # Coefficients of model used to generate data
    trueModel = np.array([1.,-1.,0.,0.,2.,0.,0.,0.25,0.])
    M = trueModel.shape[0]
    # Plotting range
    x0 = -1; x1 = 1; nx = 100; xx = np.linspace(x0,x1,nx)
    y0 = -1; y1=4

    # Sample true function at random points
    nSamples = 10
    xSamples = np.random.uniform(x0,x1,nSamples)
    for noiseSigma in [0.1,1]:
        ySamples = np.array([poly(x,trueModel) for x in xSamples])+np.random.normal(0.,noiseSigma,size=nSamples)

        # Now we wish to solve an inverse problem using this dataset. Our prior model is:
        mPrior = np.zeros_like(trueModel)
        # We construct the G matrix
        G = np.zeros([nSamples,M])
        for j in range(M):
            for i in range(nSamples):
                G[i,j] = xSamples[i]**j
        # Assume the (inverse) data covariance matrix exactly matches the true noise
        iCd = np.eye(nSamples) / noiseSigma**2
        GTG = G.T.dot(iCd).dot(G)
        GTd = G.T.dot(iCd).dot(ySamples - G.dot(mPrior))
        print "Condition number for GTG: %.3e (sigma = %.2f)"%(np.linalg.cond(GTG),noiseSigma)

        # Map out the L-curve (model norm against misfit) by performing inversion for sequence of different alpha-values
        if noiseSigma == 0.1:
            alphaMax = 5; nAlpha = 1001;
        elif noiseSigma == 1.:
            alphaMax = 10; nAlpha = 2001;
        alphaMin = 0;  aa = np.linspace(alphaMin,alphaMax,nAlpha)
        modelNorms = np.zeros_like(aa)
        modelMisfits = np.zeros_like(aa)
        for i in range(nAlpha):
            iCm = np.eye(M)*aa[i]**2
            m = np.linalg.inv(GTG+iCm).dot(GTd)
            residuals = ySamples - G.dot(mPrior) - G.dot(m)
            modelNorms[i] = m.dot(m)
            modelMisfits[i] = residuals.dot(residuals)/nSamples

        ############################
        ### Use solveLSQtikhonov ###
        ############################
        # Find optimal value of alpha and corresponding model
        mOptimalAlpha, posteriorCovariance, alpha0,beta0 = solveLSQtikhonov(GTG,GTd,fullOutput=True)
        mOptimalNorm = mOptimalAlpha.dot(mOptimalAlpha)
        residuals = ySamples - G.dot(mPrior) - G.dot(mOptimalAlpha)
        mOptimalMisfit = residuals.dot(residuals)/nSamples
        print "Optimal alpha: %.3f (sigma = %.2f)"%(alpha0,noiseSigma)
        #########################
        ### Use getLogProbTik ###
        #########################
        # Map out the distribution of P(alpha | d )
        logProbAlpha = np.array([getLogProbTik(GTG,GTd,np.eye(M),alpha,0) for alpha in aa])
        # In principle we could now just do exp(logProbAlpha) to get the probability distribution up
        # to a scale factor. However, this is likely to overflow to Inf. We therefore want to correct
        # everything relative to max(logProbAlpha) first. The result is not normalised, but it
        # is at least sensible-valued.
        probAlpha = np.exp(logProbAlpha - max(logProbAlpha))

        # Make marginal distributions for each model parameter
        m0 = -6;m1=6; mNum=2501; mm = np.linspace(m0,m1,mNum)
        marginals = np.zeros([M,mNum])
        for i in range(nAlpha):
            iCm = np.eye(M)*aa[i]**2
            pCov = np.linalg.inv(GTG+iCm)
            m = pCov.dot(GTd)
            for j in range(M):
                marginals[j,:]+=probAlpha[i]*np.exp(-(mm-m[j])**2/(2*pCov[j,j]))/np.sqrt(2*np.pi*pCov[j,j])
        marginals/=sum(probAlpha)


        ### Do the plotting ###
        plt.figure(figsize=(8,2.25))
        # Panel 1 - True model coefficients
        ax = plt.subplot(141)
        plt.bar(np.arange(M),trueModel,color='gray',linewidth=1,edgecolor='black')
        plt.ylim(-1.5,2.5)
        plt.xticks(np.arange(M)); plt.yticks(np.arange(-1,3))
        plt.xlabel("Model coeff.")
        plt.text(0.05,0.9,"(a)",transform=ax.transAxes)
        # Panel 2 - Function and data
        ax = plt.subplot(142)
        plt.plot(xx,np.array([poly(x,trueModel) for x in xx]),'k')
        plt.plot(xSamples,ySamples,'o',markerfacecolor='firebrick',markeredgecolor='black')
        plt.xlim(x0,x1); plt.ylim(y0,y1)
        plt.xticks([-1,0,1]);plt.yticks([0,2,4])
        plt.xlabel("x"); plt.ylabel("f(x)")
        plt.text(0.05,0.9,"(b)",transform=ax.transAxes)
        # Panel 3 - P(alpha | d)
        ax = plt.subplot(143)
        plt.plot(aa,probAlpha,'k')
        plt.ylim(0,1.2)
        plt.xticks(np.arange(alphaMax+1));plt.yticks([])
        plt.xlabel(r"$\alpha$"); plt.ylabel(r"$\mathbb{P}[\alpha | \mathbf{d_0}]$")
        plt.text(0.05,0.9,"(c)",transform=ax.transAxes)
        # Panel 4 - L-curve
        ax = plt.subplot(144)
        plt.plot(mOptimalMisfit,mOptimalNorm,'o',markeredgecolor='firebrick',markeredgewidth=2,fillstyle='none',markersize=6)
        plt.plot(modelMisfits,modelNorms,'k')
        plt.plot(modelMisfits[::200],modelNorms[::200],'.',color='black')
        plt.plot()
        # Axes need to be chosen for each noise level
        # Also annotate the alpha dots
        if noiseSigma == 0.1:
            plt.xlim(0,0.01); plt.ylim(2,4)
            plt.xticks([0,0.005,0.01]); plt.yticks([2,3,4])
            plt.text(0.0045,3.2,r"$\alpha=1$")
            #plt.text(0.0065,2.65,r"$\alpha=4$")
            plt.text(0.007,2.1,r"$\alpha=5$")
            plt.plot([modelMisfits[200],0.004],[modelNorms[200],3.2],'k')
            plt.plot([modelMisfits[1000],0.0085],[modelNorms[1000],2.3],'k')
        elif noiseSigma == 1:
            plt.xlim(0.0,1.8); plt.ylim(0,6)
            plt.xticks([0,0.5,1.,1.5]); plt.yticks([0,3,6])
            plt.text(0.75,1.75,r"$\alpha=1$")
            plt.text(1.25,0.6,r"$\alpha=5$")
            plt.plot([modelMisfits[200],0.7],[modelNorms[200],1.7],'k')
            plt.plot([modelMisfits[1000],1.25],[modelNorms[1000],0.5],'k')
        plt.xlabel("Mean-square resid."); plt.ylabel("Model norm")
        plt.text(0.05,0.9,"(d)",transform=ax.transAxes)
        plt.tight_layout()
        if saveFigures:
            if noiseSigma == 0.1:
                plt.savefig("prob_alpha_noise_lo.pdf")
            elif noiseSigma == 1.0:
                plt.savefig("prob_alpha_noise_hi.pdf")
        plt.show()
        # Plot posterior marginals
        plt.figure()
        for i in range(0,M):
            ax = plt.subplot(3,3,i+1)
            plt.plot(np.array([trueModel[i],trueModel[i]]),(0,1.2),'firebrick')
            # Normalise everything properly, but then rescale so that 'proper' marginal
            # has maximum value 1
            plt.plot(mm, marginals[i,:]/max(marginals[i,:]),'k')
            plt.plot(mm, np.exp(-(mm-mOptimalAlpha[i])**2/(2*posteriorCovariance[i,i]))/np.sqrt(2*np.pi*posteriorCovariance[i,i])* \
                                (sum(marginals[i,:])*(mm[1]-mm[0]))/max(marginals[i,:]),'royalblue',linestyle='dashed')
            plt.xlim(-1.5,2.5); plt.ylim(0,1.2)
            if i == 0:
                plt.ylabel(r"$\mathbb{P}[m_i \,|\, \mathbf{d_0}]$")
            if i == 8:
                plt.xticks([-1,0,1,2])
                plt.xlabel(r"$m_i$")
            else:
                plt.xticks([-1,0,1,2],[])
            plt.yticks([])
            plt.text(0.05,0.85,r"$m_%s$"%i,transform=ax.transAxes)
        plt.tight_layout()
        if saveFigures:
            if noiseSigma == 0.1:
                plt.savefig("marginals_lo.pdf")
            elif noiseSigma == 1.0:
                plt.savefig("marginals_hi.pdf")
        plt.show()
        # Now do ARD version
        #########################
        ### Use findOptimalXi ###
        #########################
        # First create xi2cov function to generate covariance matrices from a xi-vector
        def xi2cov(xi):
            nXi = xi.shape[0]
            iCm = np.diag(xi**2)
            Cm = np.diag(1./xi**2)
            diCm = np.zeros([nXi,nXi,nXi])
            for i in range(0,nXi):
                diCm[i,i,i] = 2*xi[i]
            return Cm, iCm,diCm
        # Now find optimum
        xi0 = np.ones(M) # Start with something simple
        xiOpt,info = findOptimalXi(GTG,GTd,xi0,xi2cov,bounds=M*[(1e-12,1e5)])
        print xiOpt
        #pCovXi = np.linalg.inv(GTG+np.diag(1./xiOpt))
        pCovXi = np.linalg.inv(GTG+np.diag(xiOpt))
        mXiOpt = pCovXi.dot(GTd)
        # Want to map transects through P(xi | d) centred on this point. Simplest to
        # wrap this into a function:
        def transect(icomp,xiLo,xiUp,nXi):
            result = np.zeros(nXi)
            # Avoid calling xi2cov since making iCm is easy
            iCm = np.diag(xiOpt**2)
            xiCand = np.linspace(xiLo,xiUp,nXi)
            for i in range(0,nXi):
                iCm[icomp,icomp] = xiCand[i]**2
                result[i] = getLogProbGeneral(GTG,GTd,iCm)
            return result
        # Make a plot for each component
        plt.figure()
        for icomp in range(0,M):
            ax = plt.subplot(3,3,icomp+1)
            tr = transect(icomp,1e-12,8,1000)
            plt.plot(np.linspace(1e-12,8,1000),np.exp(tr-max(tr)),'k')
            plt.text(0.05,0.05,r"$\theta_%i$"%(icomp),transform=ax.transAxes)
            plt.xlim(0,8);plt.ylim(0,1.2)
            if icomp == 0:
                plt.ylabel(r"$\mathbb{P}[\theta_i\,|\,\mathbf{d_0}]$")
            if icomp==8:
                plt.xticks([0,4,8])
                plt.xlabel(r"$\theta_i$")
            else:
                plt.xticks([0,4,8],[])
            plt.yticks([])
        plt.tight_layout()
        if saveFigures:
            if noiseSigma==0.1:
                plt.savefig("pxi_ard_lo.pdf")
            elif noiseSigma ==1:
                plt.savefig("pxi_ard_hi.pdf")
        plt.show()

        plt.figure()
        for icomp in range(0,M):
            ax = plt.subplot(3,3,icomp+1)
            plt.plot(np.array([trueModel[icomp],trueModel[icomp]]),(0,1.2),'firebrick')
            plt.plot(mm,np.exp(-(mm-mXiOpt[icomp])**2/(2*pCovXi[icomp,icomp])),'royalblue')
            plt.xlim(-1.5,2.5); plt.ylim(0,1.2)
            if icomp == 0:
                plt.ylabel(r"$\mathbb{P}[m_i \,|\, \mathbf{d_0}]$")
            if icomp == 8:
                plt.xticks([-1,0,1,2])
                plt.xlabel(r"$m_i$")
            else:
                plt.xticks([-1,0,1,2],[])
            plt.yticks([])
        plt.tight_layout()
        if saveFigures:
            if noiseSigma == 0.1:
                plt.savefig("marg_ard_lo.pdf")
            elif noiseSigma == 1.:
                plt.savefig("marg_ard_hi.pdf")
        plt.show()
    plt.figure(figsize=(4,3))
    marg = np.zeros(mNum)
    for a in np.linspace(1e-12,10,1000):
        marg+= np.exp(-0.5*(a*mm)**2)*a/np.sqrt(2*np.pi)
    plt.plot(mm,np.exp(-25*(mm**2))*np.sqrt(25/np.pi),'k--')
    plt.plot(mm,marg/(sum(marg)*(mm[1]-mm[0])),'firebrick')

    plt.xlim(-2,2);
    plt.xticks([-3,0,3]); plt.yticks([])
    plt.xlabel(r"$m_i$"); plt.ylabel(r"$\mathbb{P}[m_i]$")
    plt.tight_layout()
    if saveFigures:
        plt.savefig("prior.pdf")
    plt.show()
