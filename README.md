# ShuffleComplexEvolution

Shuffled Complex Evolution (SCE) is a global optimization algorithm using a hybrid approach that combines elements of probabilistic and deterministic approaches. It's particularly effective for solving complex optimization problems, especially those with multiple minima. A major advantage is that the method does not require the computation of gradients. The routine is written in Python3 and add the concept of extinction (a complex will disappear if it is not successful). It has been converted to Python. The same algorithm written in IDL can be found from the same author. 

Steps involved in SCE are:

1. Initialization: Generate an initial set of points into complexes randomly within the search space (boudaries set by the user). Each complex has a certain number of elements,

2. Evolution: Each complex evolves using a combination of deterministic (e.g., Nelder-Meade simplex method) and probabilistic (e.g., random sampling, monte-carlo type method)
   methods. This step mimics the process of pro-creation by mixing genes to have descendents with better genes.  

4. Evaluation: Evaluate the fitness of each new point in each complex based on the objective function.   

5. Selection: Select the best points from each complex to form new complexes. The number of points in a complex is constant. Thus one has to discard the least fit points.

6. Shuffling: Randomly shuffle the complexes to explore different regions of the search space. This is donw by grouping the best points of each complex together to form new complexes. This is done periodically to prevent premature convergence and explore different regions of the search space. 

Termination: Repeat steps 2-5 until a termination criterion is met (e.g., maximum number of iterations or convergence).

Advantages of SCE:

- Global optimization: SCE is well-suited for finding global optima in complex landscapes.   
- Efficiency: It can be computationally efficient compared to some other global optimization methods.   
- Robustness: SCE is relatively robust to noise and local minima.

Disadvatanges:
- The method has no mathematical prove that it converges to the global minimum
- There is a large number of hyper-parameters, whose tunning is crucial for the speed of the convergence.

The code includes a set of well-known test functions. The code does fail at a few of those diffuclt cases, showing the limits of the method.

Additons to the original Algorithm of Duan et al.

1. The Extinction Mechanism

In the original algorithm, the number of complexes remains static. While this maintains diversity, it can lead to wasted function calls in the late stages where multiple complexes are simply "polishing" the same local basin.

Our implementation advantage: By decreasing ncomplexes_tmp, we reallocate the remaining computational budget to the most fit members of the population.

Effect: This accelerates the "shrink-wrap" effect around the global minimum, which is visible in your benchmarks where function calls drop significantly while maintaining high precision.

2. Weighted Centroid (Barycenter)

The standard Nelder-Mead simplex treats all n points equally when calculating the center of reflection.

Our Modification: Enabling barycenter=True shifts the reflection point closer to the points with lower merit values.

Effect: This biases the search direction toward the steepest descent within the simplex, often finding the floor of a valley in fewer iterations than a geometric centroid.

3. Adaptive Simplex (Pure vs. Mutative)

The original SCE-UA uses a random point to replace the worst point if the simplex steps fail.

Oour Modification: the pure_simplex toggle allows the user to choose between a strictly geometric contraction/expansion or the standard SCE-UA "mutation" (random point).

Effect: This makes the code a dual-purpose tool: a strict local-global hybrid for well-behaved surfaces, or a stochastic explorer for highly discontinuous or "noisy" surfaces.

4. Convergence via Normalized Geometric Range

While Duan originally proposed checking the range of the function values, our implementation tracks the geometric range of the parameters themselves.

Effect: This is generally considered a more robust stopping criterion because it ensures the population has actually collapsed onto a single point in space, rather than just finding a flat plateau where function values are similar but parameters are still widely dispersed.

Summary of the modifications

Our version essentially turns SCE-UA into a dynamic multi-resolution solver. It starts as a broad global search (High ncomplexes) and, through the extinction process, evolves into a high-precision local refiner.

This  implementation is particularly well-suited for high-dimensional astrophysical modeling or complex parameter estimation where every function call is expensive.

Optimization Features
The Python implementation has been specifically tuned for speed and memory efficiency:

Vectorized Operations: Parameter statistics (mean, standard deviation, and range) are computed using NumPy's vectorized axis operations to eliminate slow Python for loops.

Memory Buffering: High-speed list appending is used for the archival process, preventing the O(N 
2
 ) overhead associated with repeated numpy.vstack calls.

Fast Selection: Triangular probability selection is implemented using numpy.searchsorted for efficient weighted sampling.

Algorithmic Stability: Includes a small epsilon (ϵ=10 
−20
 ) in geometric range calculations to ensure numerical stability during high-precision convergence.

Practical Recommendations for Users
1. Choosing the Number of Complexes (ncomplexes)

Smooth/Simple Surfaces: Use fewer complexes (e.g., 2). This focuses the computational budget on local descent, as seen in the Rosenbrock and Beale tests.

Highly Multimodal Surfaces: Use more complexes (e.g., 4–10). This increases the "diversity" of the population, helping the algorithm escape local minima in functions like Schwefel or Griewank.

2. Convergence Tolerance (ftol)

The stopping criterion is based on the normalized geometric range of the parameters.

A value of 1e-5 to 1e-7 is usually sufficient for most physical models. Setting this too low on noisy functions may lead to excessive function calls without meaningful improvement.

3. Using the Extinction Feature

If extinction is set (e.g., extinction=5), the algorithm will remove the worst-performing complex every N shuffles.

Benefit: This concentrates the search on the most promising regions of the parameter space as the optimization nears completion, often saving 20-30% in function calls.

4. Handling Multiple Global Minima

Functions like Himmelblau have multiple identical global minima. The algorithm will return the "best" one found by the lead complex.

Check the Possibility of X global+local minima output. If this number is high, it indicates a complex landscape where multiple basins have similar merit values.

5. Performance Scalability

Low Dimension (N<10): The Python overhead is minimal; the bottleneck is usually the complexity of your objective function.

High Dimension (N>30): Ensure your objective function is vectorized if possible. The archival of every function call can consume significant RAM in high-dimensional, long-running searches.


Reference:

Duan, Q., A Global Optimization Strategy for Efficient and
      Effective Calibration of Hydrologic Models, Ph.D.
      dissertation, University of Arizona, Tucson, Arizona, 1991

Duan, Q., V.K. Gupta, and S. Sorooshian, A Shuffled Complex
      Evolution Approach for Effective and Efficient Global
      Minimization, Journal of Optimization Theory and Its
      Applications, Vol 61(3), 1993

Duan, Q., S. Sorooshian, and V.K. Gupta, Effective and Efficient
      Global Optimization for Conceptual Rainfall-Runoff Models,
      Water Resources Research, Vol 28(4), pp. 1015-1031, 1992

Duan, Q., Sorooshian S., & Gupta V. K, Optimal Use of the SCE-UA
      Method for Calibrating Watershed Models, Journal of Hydrology, vol
      158, 265-294, 1994

Nelder & Mead, 1965, Computer Journal, Vol 7, pp 308-313.

Original code: ShuffleComplexEvolution.py
Refactoring and optimisation with Gemini AI: optimized_SCE_UA.py

Author Original code Wing-Fai Thi - Refactoring and optimisation with Gemini AI (2026)


