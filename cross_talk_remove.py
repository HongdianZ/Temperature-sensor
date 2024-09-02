"""

Copyright 2019, 2020 Carlos Medrano-Sanchez (ctmedra@unizar.es)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
import scipy as sc
import scipy.misc
import scipy.linalg
import scipy.interpolate
import scipy.stats
import scipy.optimize
import scipy.signal
import scipy.sparse
import time
import sys


__author__ = "Carlos Medrano-Sanchez"

__license__ = "GPL-v3"

###
def generateNodeEquations(cm, ii1):
    """
    It generates the matrix of node equations taking row ii1 as ground
    Inputs:
        cm: matrix of conductance between nodes (inverse of resistances)
        ii1: row whose voltage is considered as the reference (ground)
    Output:
        If NxM is the dimension of cm, the function returns a (N+M-1) x (N+M-1) matrix
        The order of the variables is first row voltages, then column voltages. ii1 is excluded since it is 0.
        It can be used to solve A V = b where b is a column vector of intensities coming from current sources.
    """
    N, M = cm.shape
    # N+M-1 nodes (ii1 is ground)
    # First N-1 are row wires, next M are column wires
    A = sc.zeros((N+M-1, N+M-1))
    row = 0  # Refers to A
    # Equations of horizontal wires ...
    for ii in range(N):
        if ii == ii1:
            continue
        A[row, row] = cm[ii, :].sum()
        for column in range(N-1, N+M-1):
            A[row, column] = -cm[ii, column-N+1]
        row = row + 1
    # Equations of vertical wires
    for jj in range(M):
        column = 0
        for ii in range(N):
            if ii == ii1:
                continue
            A[row, column] = -cm[ii, jj]
            column = column + 1
        A[row, row] = cm[:, jj].sum()
        row = row + 1
    return A
##
def getCeq(cm):
    """
    It generates the equivalent conductance matrix, that is the matrix
    of conductance between two nodes (one row and one column).
    Input:
        cm is the matrix of conductance obtained from the resistances that
        link directly a pair row column
    Output:
        It returns a matrix of the same dimension as cm. It is the matrix of
        equivalent conductance
    """
    A = generateNodeEquations(cm, 0)  # Row 0 is ground
    N, M = cm.shape
    # B will contain all the right part of the equations
    # that will allow to get Req between nodes
    # B is a set of column vectors, each one N+M-1
    # The first N-1 refer to row nodes (row 0 is ground so excluded)
    # and the next M elements to column nodes
    # There are N*M columns in B, corresponding to N*M pairs
    # The solution can be obtained in a single call to linalg.solve !!!
    B = sc.zeros((N+M-1, N*M))
    bcol = 0 
    for jj in range(M):
        # To obtain Req between row node 0 and column node jj
        # the source intensity goes from jj to 0
        # Since row node 0 is not present only a
        # 1 in the vector is present
        b = sc.zeros((N+M-1))
        b[N+jj-1] = 1.0
        B[:, bcol] = b[:]
        bcol = bcol + 1
    # Now the rest of pairs
    for ii in range(1, N):
        for jj in range(M):
            # Req between row node ii and column node jj
            # The source goes from node column jj (+1) 
            # and returns from node row ii (-1)
            # Note that row nodes are placed first in b vector
            b = sc.zeros((N+M-1))
            b[ii-1] = -1
            b[N+jj-1] = 1
            B[:, bcol] = b[:]
            bcol = bcol + 1
    X = sc.linalg.solve(A, B)  # X is the voltage solution for each case
    sol2 = sc.zeros((N, M))
    for jj in range(M):
        # Req between row node 0 and column node jj
        sol2[0, jj] = X[N+jj-1, jj]
    bcol = M
    for ii in range(1, N):
        for jj in range(M):
            # Req between column node jj and row node ii
            sol2[ii, jj] = X[N+jj-1, bcol] - X[ii-1, bcol]
            bcol = bcol + 1    
    return 1.0/sol2  # return equivalent conductance
###
def fixedPointSolution(cmeq, cm0=None, beta = 0.05, NITER = 25, bounds = False, ftol = 6e-6):
    """
    It finds the direct conductance of the link between each pair of 
    nodes (row-column) given the equivalent conductance of each pair.
    
    It uses a fixed point approach.
    
    Inputs:
        cmeq: equivalenteconductance matrix (measured values)
        cm0: initial guess for the solution. If None, cmeq is taken.
        beta: update parameter for iterations
        NITER: maximum number of iterations
        bounds: whether to apply bounds (force conductance to be positive)
        ftol: kind of tolerance similar toscipy.optimize.newton_krylov: f_tol : float, optional. Absolute tolerance (in max-norm) for the residual. If omitted, default is 6e-6.
    Output:
        It returns the value of the cell conductances (same dimension as cmeq)

    """

    if cm0 is None:
        c0 = cmeq
    else:
        c0 = cm0
    for nn in range(NITER):
        ceq0 = getCeq(c0)
        # Termination condition
        # Residual of the initial nonlinear problem
        residual = sc.fabs(ceq0-cmeq).max()
        print(nn+1, 'residual', residual)
        if residual < ftol:
            print(nn)
            break
        c1 = c0 - beta * (ceq0 - cmeq)
        if bounds:
            c1[c1 < 0] = 1e-18
        c0 = c1
    print(residual)
    return c1
###
def lsqrSolution(cmeq, cm0 = None, xtol = 1.49012e-08, bounds = False, ftol = 1e-8):
    """
    It finds the direct conductance of the link between each pair of 
    nodes (row-column) given the equivalent conductance of each pair.
    Use a least-square approach
    Inputs:
        cmeq is the matrix of equivalent conductances, which is the one that is measured from experiment
        cm0 is an initial value for the algorithm. If None, cmeq is taken.
        bounds boolean variable. If true a call to a function that can deal with bounds is done
        xtol, ftol: tolerance parameters for termination. They are passed to the scipy functions
    Outputs:
        It returns a matrix of the cell conductances (same dimension as cmeq)
    """
    def fobj(x, cmexp):
        # It seems that x is passed as a 1D array, so reshape
        N, M = cmexp.shape
        x2 = x.reshape(N, M)
        #x2 = x.copy()
        cm = getCeq(x2)
        return (cm-cmexp).flatten() # The square and sum is done by the optimization itself
    # xtol changes the executiom time but ftol has no clear influence
    if(cm0 is None):
        if(not bounds):
            sol = scipy.optimize.leastsq(fobj, cmeq.flatten(), cmeq, xtol = xtol, ftol = ftol)
        else:
            sol = scipy.optimize.least_squares(fun = fobj, x0 = cmeq.flatten(), xtol = xtol, bounds = (0.0, sc.inf), args = (cmeq,))
    else:
        if(not bounds):
            sol = scipy.optimize.leastsq(fobj, cm0.flatten(), cmeq, xtol = xtol, ftol = ftol)
        else:
            sol = scipy.optimize.least_squares(fun = fobj, x0 = cm0.flatten(), xtol = xtol, bounds = (0.0, sc.inf), args = (cmeq,))
    # First element of the solution is the conductance matrix itself
    N, M = cmeq.shape
    if(not bounds):
        cm = sol[0].reshape(N,M)
    else:
        cm = sol['x'].reshape(N,M)
    return cm
##
def nonlinearSolution(cmeq, cm0 = None, f_tol = None):
    """
    It finds the direct conductance of the link between each pair of 
    nodes (row-column) given the equivalent conductance of each pair.
    It uses a newton_krylov solver
    Inputs:
        cmeq is the matrix of equivalent conductances, which is the one that is measured from experiment
        cm0 is an initial value for the algorithm. If None, cmeq is taken.
        bounds boolean variable. If true a call to a function that can deal with bounds is done
        xtol, ftol: tolerance parameters for termination. They are passed to the scipy functions
    Outputs:
        It returns a matrix of the cell conductances (same dimension as cmeq)
    """
    def F(cm):
        ceq = getCeq(cm)
        return ceq-cmeq
    if(cm0 is None): cm0 = cmeq
    sol = scipy.optimize.newton_krylov(F, cm0, f_tol = f_tol)
    return sol
###
def testSolutions():
    # Random conductance matrix. Minimum value = 2e-5, max = 2e-2
    cmin = 2e-5
    cmax = 2e-2
    cm = cmin+(cmax-cmin)*sc.rand(16,16)
    # Get equivalent resistance matrix (the "measured" one)
    cmeq = get_conductances_eq(cm)
    # All the algorithms should find cm as solution
    # Newton-krylov
    cm2 = nonlinearSolution(cmeq)
    print('Newton-Krylov: Mean abs relative error ', abs((cm2-cm)/cm).mean())
    # Least squares
    cm2 = lsqrSolution(cmeq, xtol=1e-8, bounds=True)
    print('Least Squares: Mean abs relative error ', abs((cm2-cm)/cm).mean())
    # Fixed point
    cm2 = fixedPointSolution(cmeq, beta=0.1, NITER=1000, ftol=1e-12, bounds=True)
    # print('Fixed point: Mean abs relative error ', abs((cm2-cm)/cm).mean())
###

def getConductance(rawMap, Rserie = 10000.0, nbits = 12, box = None):
    """
     It obtains the conductance of a set of measurements assuming it is a voltage divider configuration. Rserie connects to Vcc. The unkown resistance is connected to ground.
     Inputs:
     rawMap: values returned by the ADC. It is usually a matrix, although the function can be used with vector inputs
     Rserie: Value of the resistance connected to Vcc
     nbits: number of bits of the ADC
     box: It is a name for the DAQ hardware used. The parameter is used to remove the effect of the multiplexer resistance and the internal resistance of the ADC.
     Ideally Rmux=0 and Rin(ADC) = infinity but sometimes they are far from ideal. If None, ideal values are taken.
     Output:
     It returns the value of the conductance.
    """
    R1 = Rserie
    lsb = 1.0/(2**nbits-1)
    val = rawMap*lsb

    val[val == 1] = 1
    val[val == 0] = 0
    res = R1*(1.0/val-1)
    if (box == 'black'):
        # Rmux is not zero and Rin(ADC) is not infinity. So take this into account
        epsilon = 1e-6
        rin = 22e3
        rmux = 300.0
        # Avoid non sense values (negative for instance)
        res[res<rmux] = rmux+epsilon
        res[res>rin] = rin-epsilon
        # Calculate mat resistance removing mux and ADC effect
        resReal = rin*res/(rin-res)-rmux
    elif (box is None):
        resReal = res
    else:
        print('box unkown')
        sys.exit(0)
    conductance = 1.0/resReal
    return conductance

