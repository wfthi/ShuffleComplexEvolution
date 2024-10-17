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

In summary:
- Global optimization: SCE is well-suited for finding global optima in complex landscapes.   
- Efficiency: It can be computationally efficient compared to some other global optimization methods.   
- Robustness: SCE is relatively robust to noise and local minima.

The code includes a set of well-known test functions. The code does fail at a few of those diffuclt cases, showing the limits of the method.

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


