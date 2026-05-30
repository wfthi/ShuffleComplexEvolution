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
           min_ncomplexes=2, max_nevolution_steps=50, alpha=1.0, beta=0.5, gamma=1.5, 
           barycenter=False, expansion=False, pure_simplex=False):
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
                                       archival_parameters, archival_merit,
                                       alpha=alpha, beta=beta, gamma=gamma,
                                       barycenter=barycenter, expansion=expansion, 
                                       pure_simplex=pure_simplex)

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
    else:
        print("Search failed")

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