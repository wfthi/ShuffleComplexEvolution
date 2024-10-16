"""
; Purpose:
;   Multidimensional minimization of a function func(x), where
;   x is an n-dimensional vector, using the Shuffled Complex
;   Evolution (SCE-UA) Optimization Method of Duan et al. with
;   some modifications.
;
; Description:
;   The SCE-UA method is an heuristic global optimization method that
;   combines features from Genetic Algorithms and Sce_Simplex Algorithms.
;   There is a high probability to find a global minimum but this
;   has not been proven mathematically.
;
;   The SCE-UA method starts with the initial selection of a
;   "population" of points distributed randomly or pseudo randomly
;   or quasi-randomly throughout the feasibe parameter space.
;   The population is then partitioned into several (ncomplexes) "complexes",
;   each containing at least 2n+1 points, where n is the number of parameters
;   to be constrained.
;    Each complex evolves independently according to a
;   "reproduction" process that, in turn, uses the Simpex Method
;   (Nelder & Mead, 1965) but without the shrinking step, which is
;   replaced by a randomly generated point.
;   At periodic stages (chosen by the user: nevolution_steps),
;   the entire population is shuffled and points are reassigned to new
;   complexes formed so that the information gained by the previous complexes
;   is shared.
;   The best "population" points in each complex are grouped into a new
;   complex. The following best points are grouped into a second complex and
;   so forth. The idea is that the best point (the local minimum) in each
;   complex will "bread" with other best points of an other complexes.
;   The "offsprings" will have characteristics (parameters) closer to the
;   best of all (closer to that set of
;   parameters that give rise to the lower merit function).
;
;   The evolution and shuffling steps continue until the prescribed
;   convergence criteria are reached.
;
;   The method combines the advantages of deterministic (Sce_Simplex)
;   and stochastic (Genetic) methods. While the genetic algorithm
;   allows global optimization, the method is slow. On the hand, the
;   Simplex (Sce_simplex) method is rapid but prompt to find local minima.
;
;   In principle other local optimizer can be used instead of the Nelder-Mead
;   Sinmplex algorithm such as Sequencial Quadratic Programming (SQP).
;
;   The performance of a global optimization solver depends on two
;   characteristics, the effectiveness and the efficiency (Duan et
;   al. 1992)
;
;   Further improvements over the original algorithm includes the
;   inclusion of the concept of "extinction".
;   The principle is the decrease of number of complexes after a
;   certain number of generations, the worst points being eliminated.
;
;   The choice of the stop criterion ftol is important and difficult
;   to make.
;
; Reference:
;
; Duan, Q., A Global Optimization Strategy for Efficient and
;      Effective Calibration of Hydrologic Models, Ph.D.
;      dissertation, University of Arizona, Tucson, Arizona, 1991
;
; Duan, Q., V.K. Gupta, and S. Sorooshian, A Shuffled Complex
;      Evolution Approach for Effective and Efficient Global
;      Minimization, Journal of Optimization Theory and Its
;      Applications, Vol 61(3), 1993
;
; Duan, Q., S. Sorooshian, and V.K. Gupta, Effective and Efficient
;      Global Optimization for Conceptual Rainfall-Runoff Models,
;      Water Resources Research, Vol 28(4), pp. 1015-1031, 1992
;
; Duan, Q., Sorooshian S., & Gupta V. K, Optimal Use of the SCE-UA
;      Method for Calibrating Watershed Models, Journal of Hydrology, vol
;      158, 265-294, 1994
;
; Nelder & Mead, 1965, Computer Journal, Vol 7, pp 308-313.
;
;
; For a Benchmark, see https://infinity77.net/global_optimization/index.html
;
; Other methods: Simplex-Simulated annealing (SIMPSA) Cardoso et al. 1996
;
; used in Thi et al. 2010 Monthly Notices of the Royal Astronomical Society,
; Volume 406, Issue 3, pp. 1409-1424
;
; Input:
;
;   ftol tolerance
;   func name of the evaluation (merit, crirterion, cost, ...) function
;   bl   lower bounds to the parameters
;   bu   upper bounds to the parameters
;   max_func_calls maximum number of function calls
;
; Optional input (keyword):
;
;   ncomplexes : number of complexes (default 2)
;   nelements_complex : number of elements per complex
;   nevolution_steps : number of evolutionary steps
;   ncalls : maximum number of function calls
;   verbose
;
; Output:
;
;   archival_parameters  all parameters set
;   archival_merit       the merit of all parameter sets considered
;
; Author:
;
;   Wing-Fai Thi (SUPA, Institut for Astronomy, Royal Observatory Edinburgh)
;   wingfai.thi at google mail adress
;
; History:
;
;   24/04/2007  European Southern Observatory (Garching, Germany)
;               Version 1.0 (IDL)
;
;   26/04/2007  Change the output to archival_parameters and
;               archival_merit take keep track of all function
;               evaluation for further parameter space statistical
;               analysis
;
;   27/02/2008  change all simplex into sce_simplex to avoid name
;               space clash
;
;   29/02/2008  add extinction: decrease of the number of complexes with
;               generations,
;               the worst points being eliminated.
;
;   01/03/2018  public version (IDL)
;
;   14/10/2024  conversion to Python3
;
; Licence: BSD
; -------------------------------------------------------------------------
"""
import numpy as np


def sce_ua(ftol, func, bl, bu, max_func_calls, seed,
           ncomplexes=2, nelements_complex=None, nevolution_steps=None,
           verbose=False, max_nshuffles=1000, extinction=None,
           min_ncomplexes=2, max_nevolution_steps=50):
    """
    Shuffled Complex Evolution (SCE-UA) optimization algorithm.

    Translated from IDL with the help of Google Gemini AI

    Args:
        ftol: foat
            Tolerance for convergence.
        func: str
            Objective function to be minimized.
        bl: array-like
            Lower bounds for parameters.
        bu: array-like
            Upper bounds for parameters.
        max_func_calls: int
            Maximum number of function calls.
        seed: int
            Random seed.
        ncomplexes: int
            Number of complexes.
        nelements_complex: int
            Number of elements in each complex.
        nevolution_steps: int
            Number of evolution steps for each complex.
        verbose: bool, optional, default=False
            Verbosity flag.
        max_nshuffles: int
            Maximum number of shuffles.
        extinction: int, optional, default=None
            Extinction parameter.
        min_ncomplexes: int, optional, default=2
            Minimum number of complexes.
        max_nevolution_steps: int, optional, default=50
            Maximum number of evolution steps.

    Returns:
        archival_parameters:
            List of archival parameters.
        archival_merit:
            List of archival merit values.
    """
    # Initialize SCE parameters
    nb_minima = 1

    # Set the seed to a specific value
    np.random.seed(seed)

    nbu = len(bu)
    nbl = len(bl)
    if nbu != nbl:
        raise ValueError("Error in input boundaries")
    nparameters = nbu
    gnrng = np.zeros(nparameters)
    bound = np.subtract(bu, bl)
    nshuffle = 0

    # Check parameter bounds
    for i in range(nparameters):
        if bound[i] < 0.0:
            raise ValueError(f"Parameter #{i}. Upper bound lower than lower bound. Please check!")

    # Select number of complexes and elements
    if min_ncomplexes is None:
        min_ncomplexes = 2
    if ncomplexes is None:
        ncomplexes = 2
    if ncomplexes < 1:
        ncomplexes = 2
    print("Number of complexes =", ncomplexes)

    if nelements_complex is None:
        nelements_complex = 2 * nparameters + 1
    if nelements_complex < (2 * nparameters + 1):
        nelements_complex = 2 * nparameters + 1
    print("Number of elements in a complex =", nelements_complex)

    # Total number of elements
    nsample = ncomplexes * nelements_complex

    # Number of evolution steps
    if max_nevolution_steps is None:
        max_nevolution_steps = 50
    if nevolution_steps is None:
        nevolution_steps = nelements_complex
    if nevolution_steps < nelements_complex:
        nevolution_steps = nelements_complex

    # Number of members in a simplex
    nelements_sce_simplex = nparameters + 1
    sce_simplex = np.zeros((nelements_sce_simplex, nparameters))
    merit_sce_simplex = np.zeros(nelements_sce_simplex)

    # Generate sample
    # Sample nsample points in the feasible parameter
    # space and compute the criterion value (merit function) at each
    # point.  In the absence of prior information, use a uniform probability
    # distribution to generate a sample.

    parameters = np.zeros((nsample, nparameters))
    archival_parameters = np.zeros((nsample, nparameters))
    for i in range(nsample):
        parameters[i] = bl + np.random.rand(nparameters) * bound
        archival_parameters[i] = parameters[i]

    nloop = 0
    ncalls = 0
    merit = np.zeros(nsample)
    archival_merit = np.zeros(nsample)
    for i in range(nsample):
        merit[i] = func(parameters[i])
        ncalls += 1
    archival_merit = merit

    # Sort: mininization
    idx = np.argsort(merit)
    parameters = parameters[idx]
    merit = merit[idx]

    # Record best and worst points. Lower merits are better
    best_parameters = parameters[0]
    best_merit = merit[0]

    # Define complexes
    cmplx = np.zeros((nelements_complex, nparameters))
    merit_cmplx = np.zeros(nelements_complex)

    # Assign a triangular probability distribution
    m = nelements_complex
    proba = 2.0 * (m - np.arange(nelements_complex)) / m / (m + 1.0)
    proba_range = np.zeros(nelements_complex + 1)
    for i in range(1, nelements_complex + 1):
        proba_range[i] = np.sum(proba[:i])

    # Define boundaries for complexes
    bl_cmplx = np.zeros(nparameters)
    bu_cmplx = np.zeros(nparameters)
    xnstd = np.zeros(nparameters)

    # Start search loop
    success = 0
    ncomplexes_tmp = ncomplexes

    while (ncalls < max_func_calls and
           success == 0 and nshuffle < max_nshuffles):
        nloop += 1
        nshuffle += 1

        # Loop over the complexes
        for icomp in range(ncomplexes_tmp):
            # Partition into complexes
            # Partition the nsample points into ncomplexes
            # complexes, each containing nelements_complex points.
            # The complexes are partitioned in such a way that the first
            # complex contains
            # every ncomplexes*k+1 ranked points, the second complex contains
            # every ncomplexes*k+2 ranked points, and so on, where k =
            # 0,2,...,nelements_complex-1.
            k1 = np.arange(nelements_complex)
            k2 = k1 * ncomplexes + icomp
            cmplx[k1] = parameters[k2]
            merit_cmplx = merit[k2]
            for i in range(nparameters):
                ave_cmp = np.mean(cmplx[k1, i])
                sig_cmp = np.std(cmplx[k1, i])
                bl_cmplx[i] = max(ave_cmp - 2 * sig_cmp, bl[i])
                bu_cmplx[i] = min(ave_cmp + 2 * sig_cmp, bu[i])

            # Competitive Evolution of Simplexes
            for evol_step in range(nevolution_steps):
                # Print generation information (optional)
                # print(f"Generation {evol_step + 1} / {nevolution_steps}")

                # Select randomly the simplex by sampling the complex
                # according to a linear probability distribution
                selected = np.full(nelements_sce_simplex, -1)
                nselected = 0
                while nselected < nelements_sce_simplex:
                    rand = np.random.rand()
                    w = np.where(rand > proba_range)[0]
                    candidate = np.max(w)
                    wchosen = np.where(candidate == selected)[0]
                    # Select a candidate that has not been chosen yet
                    if len(wchosen) == 0:
                        selected[nselected] = candidate
                        nselected += 1

                # Order the simplex
                merit_sce_simplex = merit_cmplx[selected]
                order = np.argsort(merit_sce_simplex)
                selected_order = selected[order]
                sce_simplex[np.arange(nelements_sce_simplex)] =\
                    cmplx[selected_order]
                merit_sce_simplex = merit_sce_simplex[order]

                # Generate a new sce_simplex point
                sce_simplex_new, merit_new, ncalls =\
                    generate_offspring(func, sce_simplex, merit_sce_simplex,
                                       bl, bu, bl_cmplx, bu_cmplx, ncalls,
                                       archival_parameters, archival_merit)

                # Replace the sce_simplex into the complex
                cmplx[selected_order] = sce_simplex_new
                merit_cmplx[selected_order] = merit_new

                # Sort the complex
                order = np.argsort(merit_cmplx)
                cmplx[k1] = cmplx[order]
                merit_cmplx = merit_cmplx[order]
                # end of inner loop for Competitive Evolution of Sce_Simplexes

            # Replace the complex back into the population
            parameters[k2] = cmplx[k1]
            merit[k2] = merit_cmplx
            # end of loop on Complex Evolution

        # Shuffle/Rank points
        # Sort the nsample points in order of increasing
        # criterion value so that the first point represents the point
        # with the smallest criterion value (best) and the last point
        # represents the point with the largest criterion value (worst).

        idx = np.argsort(merit)
        parameters = parameters[idx]
        merit = merit[idx]

        # Record the best and worst points
        nsample = nelements_complex * ncomplexes_tmp

        best_parameters = parameters[0]
        best_merit = merit[0]
        # worst_parameters = parameters[nsample - 1]
        # worst_merit = merit[nsample - 1]

        # Compute the standard deviation for each parameter xnstd
        # and the normalized geometric range of the parameters gnrng
        for i in range(nparameters):
            xnstd[i] = np.std(parameters[:, i])
            gnrng[i] = np.exp(np.mean(np.log((np.max(parameters[:, i])
                                              - np.min(parameters[:, i]))
                                      / bound[i])))

        # Define the criterion for finding the optimum
        # There are alternative stopping criterion
        # theses criteria may not be appropriate for the step function test
        crit = np.min(gnrng)

        # seems to work better with step, not with Schwefel, Corana
        # crit = np.median(gnrng)  # median instead of mean

        if nshuffle > 1:
            # crit = abs(previous_best_merit - best_merit)
            if crit > 0.0 and crit < ftol:
                success = 1

        # previous_best_merit = best_merit

        # Extinction
        # If this option is chosen, the complex with the worst points is
        # eliminated after "extinction" shuffling
        # steps until min_ncomplexes are left.
        if extinction is not None and ncomplexes_tmp > min_ncomplexes:
            if nshuffle % extinction == 0:
                ncomplexes_tmp -= 1

        # Increase the number of evolutionary steps
        # if there is only one complex left
        if ncomplexes_tmp == 1:
            nevolution_steps = max_nevolution_steps

        # Verbose
        if verbose:
            print()
            print("shuffle =", nshuffle)
            print("number of complexes =", ncomplexes_tmp)
            print("Number of function calls =", ncalls, " best =",
                  best_merit, " crit =", crit)
            print("best parameters =", best_parameters)

        # end of the search

    best_parameters = parameters[0]
    best_merit = merit[0]

    # Possibility of multi minima
    nbest = ncomplexes_tmp
    for i in range(1, nbest):
        test = np.abs(merit[i] - merit[0]) / (1e30 + merit[0])
        if test < 1e-3:
            # Possible multi global minima
            nb_minima += 1

    # Display the results
    if success == 1:
        print()
        print("Convergence criterion :", crit)
        print()
        print("number of function calls:", ncalls)
        print()
        print("Results")
        print("Parameters:")
        print(best_parameters)
        print("Evaluation function =", best_merit)
        if nb_minima > 1:
            print("Possibility of", nb_minima, " global+local minima")

        if ncalls >= max_func_calls:
            print("Optimization search terminated because the limit")
            print("on the maximum number of trials")
            print(max_func_calls)
            print("has beem exceeded. Search was stopped at trial number:")
            print(ncalls)
            print("of the initial loop!")

    # Sort archival merit
    jsort = np.argsort(archival_merit)
    archival_parameters = archival_parameters[jsort]
    archival_merit = archival_merit[jsort]

    return archival_parameters, archival_merit


def generate_offspring(func, s, sf, bl, bu, bl_cmplx, bu_cmplx,
                       ncalls, archival_parameters, archival_merit,
                       alpha=1.0, beta=0.5, gamma=1.5,
                       barycenter=False, expansion=False, pure_simplex=False):
    """
    Generates a new point in a simplex.

    Args:
        func: str
            Objective function.
        s: Sorted simplex.
        sf: Function values.
        bl: array of floats
            Lower bounds.
        bu: array of floats
            Upper bounds.
        bl_cmplx:
            Lower bounds for complex.
        bu_cmplx:
            Upper bounds for complex.
        ncalls: int
            Number of function calls.
        archival_parameters:
            Archival parameters.
        archival_merit:
            Archival merit.
        alpha: float, optional, default=1.0
            Reflection coefficient.
        beta: float, optional, default=0.5
            Contraction coefficient.
        gamma: float, optional, default=1.5
            Expansion coefficient.
        barycenter:  bool, optional, default=False
            Use weighted centroid.
        expansion: bool, optional, default=False
            Use expansion step.
        pure_simplex: bool, optional, default=False
            Use pure simplex method.

    Returns:
        snew: New point.
        fnew: Function value at new point.
        ncalls : updated number of function calls
    """

    dimension = s.shape
    nparameters = dimension[1]

    # Use default Nelder-Mead coefficients if not specified
    alpha = alpha if alpha is not None else 1.0
    beta = beta if beta is not None else 0.5
    gamma = gamma if gamma is not None else 1.5

    # Assign the best and worst points
    sw = s[-1]
    fw = sf[-1]

    # Compute centroid
    if not barycenter:
        ce = np.mean(s[:-1], axis=0)
    else:
        weights = 1.0 / sf[:-1]
        ce = np.sum(weights[:, np.newaxis] * s[:-1], axis=0) / np.sum(weights)

    fnew = fw + 100.0

    # Attempt expansion point
    if expansion and pure_simplex:
        snew = ce + gamma * (ce - sw)

        # Check bounds
        if np.all(snew >= bl) and np.all(snew <= bu):
            fnew = func(snew)
            archival_parameters = np.vstack([archival_parameters, snew])
            archival_merit = np.append(archival_merit, fnew)
            ncalls += 1

    # Attempt reflection point
    if fnew > fw:
        snew = ce + alpha * (ce - sw)

        # Check bounds
        if np.any(snew < bl) or np.any(snew > bu):
            snew = bl_cmplx + np.random.rand(nparameters) *\
                  (bu_cmplx - bl_cmplx)

        fnew = func(snew)
        archival_parameters = np.vstack([archival_parameters, snew])
        archival_merit = np.append(archival_merit, fnew)
        ncalls += 1

        # Attempt contraction point
        if fnew > fw:
            snew = sw + beta * (ce - sw)
            fnew = func(snew)
            archival_parameters = np.vstack([archival_parameters, snew])
            archival_merit = np.append(archival_merit, fnew)
            ncalls += 1

            # Both reflection and contraction failed, attempt random point
            if fnew > fw:
                if pure_simplex:
                    snew = sw - beta * (ce - sw)
                else:
                    snew = bl_cmplx + np.random.rand(nparameters) *\
                          (bu_cmplx - bl_cmplx)

                archival_parameters = np.vstack([archival_parameters, snew])
                fnew = func(snew)
                archival_merit = np.append(archival_merit, fnew)
                ncalls += 1

    return snew, fnew, ncalls


def rosenbrock(x):
    """Rosenbrock function.

    The Rosenbrock Function is unimodal
    Bound: X1=[-5,5], X2=[-2,8]
    Global Optimum: 0,at (1,1) is not easy to find because it is situated in a
    valley with a flat bottom.

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Rosenbrock function at the given point.
    """
    a = 100.0
    x1, x2 = x
    return a * (x2 - x1**2)**2 + (1.0 - x1)**2


def goldstein_price(x):
    """Goldstein-Price function.

    This is the Goldstein-Price Function
    Bound X1=[-2,2], X2=[-2,2]
    Global Optimum: 3.0,(0.0,-1.0)

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Goldstein-Price function at the given point.
    """
    x1, x2 = x
    u1 = (x1 + x2 + 1.0)**2
    u2 = 19.0 - 14.0 * x1 + 3.0 * x1**2 - 14.0 * x2
    u2 += 6.0 * x1 * x2 + 3.0 * x2**2
    u3 = (2.0 * x1 - 3.0 * x2)**2
    u4 = 18.0 - 32.0 * x1 + 12.0 * x1**2
    u4 += 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2**2
    u5 = u1 * u2
    u6 = u3 * u4
    return (1.0 + u5) * (30.0 + u6)


def camelback(x):
    """Six-hump Camelback function.

    This is the Six-hump Camelback Function.
    Bound: X1=[-3,3], X2=[-2,2]
    True Optima: -1.031628453489877, (-0.08983,0.7126), (0.08983,-0.7126)
    also called sixmin

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Six-hump Camelback function at the given point.
    """
    x1, x2 = x
    f = (4.0 - 2.1 * x1**2 + (x1**4) / 3.) * x1**2
    f += x1 * x2 + (-4.0 + 4.0 * x2**2) * x2**2
    return f


def rastrigin(x):
    """Rastrigin function.

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Rastrigin function at the given point.
    """
    x1, x2 = x
    return x1**2 + x2**2 - np.cos(18.0 * x1) - np.cos(18.0 * x2)


def two_peak_trap(x):
    """Two-peak trap function.

    The global maximum of this function is x=0

    Args:
      x: A scalar representing the input variable.

    Returns:
      The value of the Two-peak trap function at the given point.
    """
    if 0.0 <= x < 15.0:
        return 160.0 * (15.0 - x) / 15.0
    elif 15.0 <= x <= 20.0:
        return 200.0 * (x - 15.0) / 5.0
    else:
        raise ValueError("x is outside the valid range (0.0, 20.0]")


def griewank_2d(x):
    """Griewank function (2-D).
    A multimodal function with a high number of local minima and
    a single global minimum.

    This is the Griewank Function (2-D or 10-D)
    Bound: X(i)=[-600,600], for i=1,2,...,10
    Global Optimum: 0, at origin

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Griewank function at the given point.
    """
    d = 200.0
    u1 = 0.0
    u2 = 1.0
    for j in range(2):
        u1 += x[j] * x[j] / d
        u2 *= np.cos(x[j] / np.sqrt(j + 1))
    return u1 - u2 + 1.0


def griewank_10d(x):
    """Griewank function (10-D).

    Griwwabgk's function 10 dimensions
    bounds -400. < x_j < 400. j=0,...,9
    Global minimum at origin = 0.

    Args:
      x: A list or array of size 10 representing the input variables.

    Returns:
      The value of the Griewank function at the given point.
    """
    n = 10
    d = 4000.0

    u1 = 0.0
    u2 = 1.0
    for j in range(n):
        u1 += x[j] * x[j] / d
        u2 *= np.cos(x[j] / np.sqrt(j + 1))
    return u1 - u2 + 1.0


def step(x):
    """Step function.

    Storn & Price 1997
    5 numbers

    Args:
      x: A list or array of any size representing the input variables.

    Returns:
      The value of the step function at the given point.
    """
    return 25.0 + np.sum(np.floor(x))


def sphere(x):
    """Sphere function.

    bound xj = [-5.12, 5.12]
    minimum at (0.,0.,0.) fmin=0

    Args:
      x: A list or array of size 3 representing the input variables.

    Returns:
      The value of the sphere function at the given point.
    """
    return x[0]**2 + x[1]**2 + x[2]**2


def ackley(x):
    """Ackley function.

    Ackley function.
    search domain -15=< x_i =< 30
    global minimum at (0,0,...)=0.
    several local minima
    The number of variables n should be adjusted below.
    The default value of n =2.

    Args:
      x: A list or array of any size representing the input variables.

    Returns:
      The value of the Ackley function at the given point.
    """
    n = len(x)
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi

    s1 = 0.0
    s2 = 0.0
    for i in range(n):
        s1 += x[i]**2
        s2 += np.cos(c * x[i])

    return -a * np.exp(-b * np.sqrt(1.0 / n * s1)) -\
        np.exp(1.0 / n * s2) + a + np.exp(1.0)


def beale(x):
    """Beale function.

    The number of variables n = 2.
    Bounds -4.5 =< x_i =< 4.5
    Global minimum at (3.,0.5)=0.

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Beale function at the given point.
    """
    x1, x2 = x
    f = (1.5 - x1 * (1.0 - x2))**2
    f += (2.25 - x1 * (1.0 - x2**2))**2 + (2.625 - x1 * (1 - x2**3))**2
    return f


def booth(x):
    """Booth function.

    Booth function
    bounds -10 =< x_i =< 10
    several local minima
    Global minimum at (1.,3.)=0.
    The number of variables n = 2.

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Booth function at the given point.
    """
    x1, x2 = x
    return (x1 + 2.0 * x2 - 7.0)**2 + (2.0 * x1 + x2 - 5.0)**2


def schwefel(x):
    """Schwefel function.
    A multimodal function with a single global minimum and many local minima.

    bounds -500=< x_i =< 500 i=1,2,3,...,n
    several local minima
    Global minima at 420.9687d0*(1.,1.,...,1.)=0.
    The number of variables n should be adjusted below.
    The default value of n = 2.

    Args:
      x: A list or array of any size representing the input variables.

    Returns:
      The value of the Schwefel function at the given point.
    """
    n = len(x)
    s = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
    return 418.9829 * n + s


def michalewicz(x):
    """Michalewicz function.

    bounds 0 =< x_i =< pi
    several local minima
    Global minimum at
    n=2 -1.8013
    n=5  -4.4687658
    n=10 -9.66015
    The number of variables n should be adjusted below.
    The default value of n =2.

    Args:
      x: A list or array of any size representing the input variables.

    Returns:
      The value of the Michalewicz function at the given point.
    """
    n = len(x)
    m = 10.0
    s = 0.0
    for i in range(n):
        s += np.sin(x[i]) * (np.sin((i * x[i]**2.0) / np.pi))**(2.0 * m)
    return -s


def himmelblau(x):
    """
    This is the Himmelblau Function (see textbook by himmelblau)
    Bound: X1=[-5,5], X2=[-5,5]
    Global Optimum: 0 (3,2)
    https://en.wikipedia.org/wiki/Himmelblau%27s_function

    Args:
        x: A list or array of size 2 representing the input variables.

    Returns:
        The value of the Himmelblau function at the given point.
    """
    f1 = x[0] * x[0] + x[1] - 11.0
    f2 = x[0] + x[1] * x[1] - 7.0
    return f1 * f1 + f2 * f2


def goldstein(x):
    """Goldstein-Price function.

    Goldstein and Price function
    bounds -2 =< x_i =< 2 i=1,2
    several local minima
    Global minimum (0.,1.)=3.0d0
    The number of variables n = 2.

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Goldstein-Price function at the given point.
    """
    a = 1.0 + (x[0] + x[1] + 1.0)**2 * (19.0 - 14.0 * x[0] + 3.0 * x[0]**2
                                        - 14.0 * x[1] +
                                        6.0 * x[0] * x[1] + 3.0 * x[1]**2)
    b = 30.0 + (2.0 * x[0] - 3.0 * x[1])**2 * (18.0 - 32.0 * x[0] +
                                               12.0 * x[0]**2
                                               + 48.0 * x[1] -
                                               36.0 * x[0] * x[1]
                                               + 27.0 * x[1]**2)
    return a * b


def easom(x):
    """Easom function.

    Easom function
    n = 2
    bounds -10< x_i < 10 i=1,2
    several local minima
    Global minimum at (pi,pi)=-1.

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Easom function at the given point.
    """
    pi = np.pi
    return -np.cos(x[0]) * np.cos(x[1]) *\
        np.exp(-(x[0] - pi)**2 - (x[1] - pi)**2)


def branin(x):
    """Branin RCOS function.

    Branin RCOS function
    n = 2
    bounds -5<x_1<10, 0<x_2<15
    no local minimum
    Global minima at (-pi,12.275), (pi,2.275), (9.42478,2.475)

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Branin RCOS function at the given point.
    """
    pi = np.pi
    f = (x[1]
         - (5.1 / (4.0 * pi**2)) * x[0]**2
         + 5.0 * x[0] / pi
         - 6.0)**2 + 10.0 * (1.0 - 1.0 / (8.0 * pi)) * np.cos(x[0]) + 10.0
    return f


def f1_test(x):
    """Test function f1.

    n = 2
    bounds -1< x_i < 1 i=1,2
    many local minima
    Global minimum at (0.,0.)=-2

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the test function f1 at the given point.
    """
    return x[0]**2 + x[1]**2 - np.cos(18.0 * x[0]) - np.cos(18.0 * x[1])


def f2_test(x):
    """Test function f2.

    n=10
    many local minima
    bounds 0< x_i < 1 i=1,2,...,10
    Global minimum at (0.4,0.4,...,0.4)=0.

    Args:
      x: A list or array of size 10 representing the input variables.

    Returns:
      The value of the test function f2 at the given point.
    """
    n = 10
    a = 0.05
    f = 0.0
    for i in range(n):
        f += np.min([abs(x[i] - 0.2) + a, abs(x[i] - 0.4),
                     abs(x[i] - 0.7) + a])
    return f


def shubert(x):
    """Shubert function.

    bounds -10. =< x_i =< 10.
    several local minima
    Global minimum -186.7309
    The number of variables n =2.

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of the Shubert function at the given point.
    """
    s1 = 0.0
    s2 = 0.0
    for i in range(1, 6):
        i1 = float(i)
        i2 = float(i + 1)
        s1 += i1 * np.cos(i2 * x[0] + i1)
        s2 += i1 * np.cos(i2 * x[1] + i1)
    f = s1 * s2
    return f


def zakharov(x):
    """Zakharov function.

    n = 2,5,10
    bounds -5< x_i < 10 i=1,2,..., n
    no local minima
    Global minima at (0.,0.,...,0.)=0.

    Args:
        x: A list or array of any size representing the input variables.

    Returns:
        The value of the Zakharov function at the given point.
    """
    n = len(x)
    j = np.arange(1, n + 1)
    y = (np.sum(0.5 * j * x))**2
    f = np.sum(x**2) + y * (1.0 + y)
    return f


def zimmermann(x):
    """Zimmermann's function.

    Zimmerman's problem
    n=2
    bounds x_i > 0 i=1,2

    Args:
      x: A list or array of size 2 representing the input variables.

    Returns:
      The value of Zimmermann's function at the given point.
    """
    f = 9.0 - x[0] - x[1]
    p = (x[0] - 3.0)**2 + (x[1] - 2.0)**2
    q = x[0] * x[1]
    if p > 16.0 or q > 14.0:
        f = 100.0
    return f


def corana(x):
    """Corana's parabola function.
    n = 2
    bounds -1000. < < 1000.
    minimum at -0.05 < x_j < 0.05 j=0,1,2,3
    minimum = 0.

    Args:
      x: A list or array of size 4 representing the input variables.

    Returns:
      The value of Corana's parabola function at the given point.
    """
    d = [1.0, 1e3, 1e1, 1e2]
    z = np.fix(np.abs(x / 0.2) + 0.49999) * (np.fix(x > 0.0)
                                             - np.fix(x < 0.0)) * 0.2
    f = 0.0
    for i in range(4):
        if abs(x[i] - z[i]) < 0.05:
            signz = np.fix(z[i] > 0.0) - np.fix(z[i] < 0.0)
            f += 0.15 * (z[i] - 0.05 * signz)**2 * d[i]
        else:
            f += d[i] * x[i]**2
    return f


def noisy_quartic_30d(x):
    """De Jong's noisy quartic function (30 dimensions).

    bounds -1.28 =< x_j -< 1.28

    Args:
      x: A list or array of size 30 representing the input variables.

    Returns:
      The value of the noisy quartic function at the given point.
    """
    n = 30
    rn = np.random.rand()
    f = 0.0
    for i in range(n):
        f += (i + 1) * x[i]**4 * (1.0 + 2.0 * rn)
    return f


def shekel10(x):
    """Shekel function with 10 local minima.

    adapated from https://www.sfu.ca/~ssurjano Matlab version
    m local minima
    bounds 0. < (x1,x2,x3,x4) < 10. x=(x1,x2,x3,x4)
    global minimum at (4,4,4,4) f = -10.5364

    Args:
      x: A list or array of size 4 representing the input variables.

    Returns:
      The value of the Shekel function at the given point.
    """
    m = 10
    b = np.array([1., 2., 2., 4., 4., 6., 3., 7., 5., 5.]) * 0.1
    C = np.array([[4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                  [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6],
                  [4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                  [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6]]).T
    outer = 0.0
    for ii in range(m):
        bi = b[ii]
        inner = 0.0
        for jj in range(4):
            xj = x[jj]
            Cij = C[ii, jj]
            inner += (xj - Cij)**2
        outer += 1.0 / (inner + bi)
    return -outer


def styblinski_tang(x):
    """Styblinski-Tang function.

    The global minimum of the Styblinski-Tang function is at the point where
    all x_i are equal to -2.903534. The corresponding minimum value is
    -39.16599 * n, where n is the number of dimensions of the function.

    For example, in 2 dimensions, the global minimum is at
    (-2.903534, -2.903534) with a value of -78.33198

    Args:
      x: A list or array of any size representing the input variables.

    Returns:
      The value of the Styblinski-Tang function at the given point.

    # Example usage:
    x = np.array([1.0, 2.0, 3.0])
    result = styblinski_tang(x)
    print(result)  # Output: -39.16599

    """
    n = len(x)
    return np.sum([x[i]**4 - 16 * x[i]**2 + 5 * x[i] for i in range(n)]) / n


def levy2(X):
    """Levy function.
    https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective

    A multimodal function with a single global minimum and many local minima.
    The global minimum of the Levy function for n=2 is at the point (1, 1).
    The corresponding minimum value is 0.

    Args:
      X: A list or array of size 2

    Returns:
      The value of the Levy function at the given point.

    Example:
    >>> levy2(np.array([1., 1.]))  #
    """
    x, y = X
    res = (np.sin(3 * np.pi * x)**2
           + (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2)
           + (y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2))

    return res


def test_sce(seed=42141):
    """
    Test function for the SCE-UA optimization algorithm.

    Args:
        parameters: A list or array to store the optimized parameters.
        merit: A list or array to store the corresponding merit values.
    """

    ftol = 1e-6
    print()
    print("------------------------------------")
    print("Test: Himmelblau")
    _, _ = sce_ua(ftol, himmelblau, bl=[-5.0, -5.0], bu=[5.0, 5.0],
                  max_func_calls=10000, seed=seed,
                  ncomplexes=4, nevolution_steps=20)
    print("Tolerance =", ftol)
    print("Global minimum at (3.0,2.0)=0.0")
    print("+3 others minima")

    ftol = 1e-5
    print("------------------------------------")
    print("Test: Shekel m=10")
    _, _ = sce_ua(ftol, shekel10, [0.0, 0.0, 0.0, 0.0],
                  [10.0, 10.0, 10.0, 10.0],
                  10000, seed,
                  ncomplexes=4, nevolution_steps=20, extinction=1)
    print("Tolerance =", ftol)
    print("Global minimum at (4,4,4,4) = -10.5364")
    print("With extinction")

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: Rosenbrock")
    _, _ = sce_ua(ftol, rosenbrock, [-5.0, -2.0], [5.0, 8.0], 10000, seed,
                  nevolution_steps=20, ncomplexes=4)
    _, _ = sce_ua(ftol, rosenbrock, [-5.0, -2.0], [5.0, 8.0], 10000, seed,
                  nevolution_steps=20, ncomplexes=2,
                  extinction=1, max_nevolution_steps=20)
    print("Tolerance =", ftol)
    print("Global minimum at (1.0,1.0)=0.0")

    ftol = 1e-3
    print()
    print("------------------------------------")
    print("Goldstein Price")
    print("Bound X1=[-2,2], X2=[-2,2]")
    print("Global Optimum: 3.0,(0.0,-1.0)")
    _, _ = sce_ua(ftol, goldstein_price, [-2.0, -2.0], [2.0, 2.0],
                  1000, seed,
                  ncomplexes=2, nevolution_steps=20)
    _, _ = sce_ua(ftol, goldstein_price, [-2.0, -2.0], [2.0, 2.0],
                  1000, seed,
                  ncomplexes=2, nevolution_steps=20, extinction=1)

    ftol = 1e-10
    print()
    print("------------------------------------")
    print("The Rastrigin Function has many hills and valleys")
    print("Bound: X1=[-1,1], X2=[-1,1]")
    print("Global Optimum: -2, (0,0)")
    _, _ = sce_ua(ftol, rastrigin, [-1.0, -1.0], [1.0, 1.0], 10000, seed,
                  ncomplexes=4, min_ncomplexes=2)
    _, _ = sce_ua(ftol, rastrigin, [-1.0, -1.0], [1.0, 1.0], 10000, seed,
                  ncomplexes=4, extinction=1, min_ncomplexes=2)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: 6 hump camelback")
    print("Bound: X1=[-3,3], X2=[-2,2]")
    print("True Optima: -1.031628453489877, (-0.08983,0.7126), (0.08983,-0.7126)")

    _, _ = sce_ua(ftol, camelback, [-3.0, -2.0], [3.0, 2.0], 10000, seed,
                  nevolution_steps=10, ncomplexes=4)
    print()
    print("Test: 6 hump camelback")
    print("Extinction option on")
    _, _ = sce_ua(ftol, camelback, [-3.0, -2.0], [3.0, 2.0], 10000, seed,
                  nevolution_steps=10, ncomplexes=4,
                  extinction=1, min_ncomplexes=1)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: Griewank 2d")
    print("minimum at origin =0")
    _, _ = sce_ua(ftol, griewank_2d, [-600.0, -600.0], [600.0, 600.0], 10000,
                  seed,  nevolution_steps=20)

    ftol = 1e-5
    print()
    print("------------------------------------")
    print("Test: Griewank 10d")
    print("minimum at origin =0")
    bl = np.repeat(-400.0, 10)
    bu = np.repeat(400.0, 10)
    _, _ = sce_ua(ftol, griewank_10d, bl, bu, 10000, seed,
                  nevolution_steps=20, ncomplexes=2, extinction=1)

    ftol = 1e-3
    print()
    print("------------------------------------")
    print("Test: noisy quartic 30d")
    print("minimum =< 30")
    n = 30
    bl = np.repeat(-1.28, n)
    bu = np.repeat(1.28, n)
    _, _ = sce_ua(ftol, noisy_quartic_30d, bl, bu, 15000, seed,
                  nevolution_steps=20, ncomplexes=6)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: sphere")
    print("minimum at (0.,0.,0.)=0.")
    bl = [-5.12, -5.12, -5.12]
    bu = [5.12, 5.12, 5.12]
    _, _ = sce_ua(ftol, sphere, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, extinction=1)

    ftol = 1e-2
    print()
    print("------------------------------------")
    print("Test: step")
    print("minimum at (-5.,-5.,-5.,-5.,-5.)=0.")
    bl = [-5.12, -5.12, -5.12, -5.12, -5.12]
    bu = [5.12, 5.12, 5.12, 5.12, 5.12]
    _, _ = sce_ua(ftol, step, bl, bu, 10000, seed,
                  nevolution_steps=5,
                  ncomplexes=6, nelements_complex=5, extinction=4,
                  max_nevolution_steps=50)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: Ackley function")
    print("search domain -15=< x_i =< 30")
    print("global minimum at (0.,0.,)=0.")
    print("several local minima")
    bl = [-15.0, -15.0]
    bu = [30., 30.]
    _, _ = sce_ua(ftol, ackley, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, nelements_complex=5,
                  extinction=1, max_nevolution_steps=20)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: Beale function")
    print("Bounds -4.5 =< x_i =< 4.5")
    print("Global minimum at (3.,0.5)=0.")
    bl = [-4.5, -4.5]
    bu = [4.5, 4.5]
    _, _ = sce_ua(ftol, beale, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, nelements_complex=5,
                  extinction=1, max_nevolution_steps=20)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: Booth function")
    print("bounds -10 =< x_i =< 10")
    print("several local minima")
    print(" Global minimum at (1.,3.)=0.")
    bl = [-10, -10]
    bu = [10, 10]
    _, _ = sce_ua(ftol, booth, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, nelements_complex=5,
                  extinction=1, max_nevolution_steps=20)

    ftol = 1e-5
    print()
    print("------------------------------------")
    print("Test: Schwefel function")
    print("bounds -500=< x_i =< 500 i=1,2")
    print("several local minima")
    print("Global minima at (420.9687d0,420.9687d0)=0.")
    bl = [-500, -500]
    bu = [500, 500]
    _, _ = sce_ua(ftol, schwefel, bl, bu, 10000, seed,
                  nevolution_steps=10, ncomplexes=4, nelements_complex=5,
                  max_nevolution_steps=30)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: Goldstein Price function")
    print("bounds -1 =< x_i =< 2 ")
    print("several local minima")
    print("Global minimum at   (0.,-1.)=3.0")
    bl = np.repeat(-1.0, 2)
    bu = np.repeat(2.0, 2)
    _, _ = sce_ua(ftol, goldstein, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, nelements_complex=5,
                  extinction=1,
                  max_nevolution_steps=30)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: Shubert function")
    print("bounds -10. =< x_i =< 10.")
    print("several local minima")
    print("Global minimum -186.7309")
    bl = np.repeat(-10.0, 2)
    bu = np.repeat(10.0, 2)
    _, _ = sce_ua(ftol, shubert, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, nelements_complex=5,
                  extinction=2,
                  max_nevolution_steps=30)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Easom function")
    print("bounds -10. < x_i < 10.")
    print("several local minima")
    print("Global minimum (pi,pi)=-1.")
    bl = np.repeat(-10.0, 2)
    bu = np.repeat(10.0, 2)
    _, _ = sce_ua(ftol, easom, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, nelements_complex=5,
                  max_nevolution_steps=30)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("f2_test function, n=10")
    print("bounds 0. < x_i < 1.")
    print("several local minima")
    print("Global minimum (0.4,0.4,...,0.4)=0.")
    bl = np.repeat(0.0, 10)
    bu = np.repeat(1.0, 10)
    _, _ = sce_ua(ftol, f2_test, bl, bu, 15000, seed,
                  ncomplexes=2, extinction=2)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: Zakharov function, n=2")
    print("bounds -5. < x_i < 10.")
    print("no local minimum")
    print("Global minimum (0.,0.,...,0.)=0.")
    bl = np.repeat(-5.0, 2)
    bu = np.repeat(10.0, 2)
    _, _ = sce_ua(ftol, zakharov, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, nelements_complex=5,
                  max_nevolution_steps=20)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: Branin function, n=2")
    print("bounds -5. < x_1 < 10., 0.<x_2<15  ")
    print("no local minimum")
    print("Global minimum at (-pi,12.275), (pi,2.275), (9.42478,2.475)")
    bl = [-5.0, 0.0]
    bu = [10.0, 15.0]
    _, _ = sce_ua(ftol, branin, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, nelements_complex=5,
                  max_nevolution_steps=20)

    ftol = 1e-4
    print()
    print("------------------------------------")
    print("Test: f1_test function, n=2")
    print("bounds -1. < x_1 < 1.")
    print("manu local minima")
    print("Global minimum at (0.,0.)=-2.")
    bl = [-1.0, -1.0]
    bu = [1.0, 1.0]
    _, _ = sce_ua(ftol, f1_test, bl, bu, 10000, seed,
                  nevolution_steps=5, ncomplexes=2, nelements_complex=5,
                  max_nevolution_steps=20)

    ftol = 1e-5
    print()
    print("------------------------------------")
    print("Test: Zimmermann''s problem, n=2")
    print("bounds x_i >0. i=1,2")
    print("local minima?")
    print("Global minimum at (7.,2.)=0.")
    bl = [0.0, 0.0]
    bu = [100.0, 100.0]
    _, _ = sce_ua(ftol, zimmermann, bl, bu, 10000, seed,
                  ncomplexes=5)

    ftol = 1e-5
    print()
    print("------------------------------------")
    print("Test: Corana''s parabola, n=4")
    print("bounds -1000. < x_i < 1000. i=0,1,2,3")
    print("many local minima")
    print("Global minimum at abs(x_j)<0.05")
    bl = np.repeat(-1000.0, 4)
    bu = np.repeat(1000.0, 4)
    _, _ = sce_ua(ftol, corana, bl, bu, 50000, seed,
                  nevolution_steps=5, ncomplexes=5, nelements_complex=10,
                  max_nevolution_steps=20)

    ftol = 1e-6
    print()
    print("------------------------------------")
    print("Styblinski-Tang n=2")  # suggested by Google Gemini AI
    n = 2
    max_func_calls = 10000
    bl = np.repeat(-10., n)
    bu = np.repeat(10., n)
    print("global minimum is at (-2.903534, -2.903534) with -78.33198)")
    _, _ = sce_ua(ftol, styblinski_tang, bl, bu, max_func_calls,
                  seed, ncomplexes=5)

    ftol = 1e-6
    print()
    print("------------------------------------")
    print("Levy n=2")  # suggested by Google Gemini AI
    n = 2
    max_func_calls = 10000
    bl = [-10, -10]
    bu = [10, 10]
    print("global minimum is at (1, 1) with 0)")
    _, _ = sce_ua(ftol, levy2, bl, bu, max_func_calls,
                  seed, ncomplexes=5)


if __name__ == "__main__":
    # Call the SCE-UA algorithm
    print('Test: Himmelblau')
    archival_parameters, archival_merit\
        = sce_ua(ftol=1e-6,
                 func=himmelblau,
                 bl=[-5., -5.],
                 bu=[5., 5.],
                 max_func_calls=10000,
                 ncomplexes=4,
                 nevolution_steps=20,
                 verbose=False,
                 seed=2)
    #
    print()
    print('Test: Rastrigin')
    # Global Optimum: -2, (0,0)
    archival_parameters, archival_merit\
        = sce_ua(ftol=1e-6,
                 func=rastrigin,
                 bl=[-1., -1.],
                 bu=[1., 1.],
                 max_func_calls=10000,
                 ncomplexes=4,
                 nevolution_steps=20,
                 verbose=False,
                 seed=2)
    #
    print()
    print('Camel back')
    _, _ = sce_ua(1e-4, camelback, [-3.0, -2.0], [3.0, 2.0], 10000, 42,
                  nevolution_steps=10, ncomplexes=4)

    #
    # print()
    # print('Zimmermann')
    # _, _ = sce_ua(1e-4, zimmermann, [0.0, 0.0], [100.0, 100.0], 10000, 423,
    #               nevolution_steps=5, ncomplexes=10,
    #               nelements_complex=5,
    #               max_nevolution_steps=20,
    #               extinction=1)

    test_sce()
