import numpy as np
import numba

import matplotlib.pyplot as plt

def capacityDimension(points, minL=0.01, maxL=0.2, sizeSteps=30, debug=False):
    """    
    Compute the capacity (Hausdorf-Besicovich) dimension
    of a set of points using the box-counting algorithm.

    Computation scales linearly with the number of points.

    Runs either natively in python (to produce debug plots)
    or in C using numba depending on the value of the `debug`
    keyword argument and if the data is 2D or not.
    
    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Points to calculate the correlation dimension of.

    minL : float
        The minimum value of the box size to check, given in
        units of the total size of the point cloud.
        
    maxL : float
        The maximum value of the box size to check, given in
        units of the total size of the point cloud.

    sizeSteps : int
        The number of box sizes to use in the provided range
        to compute the capacity dimension.

    debug : bool
        Whether to plot diagnostic information.

    Returns
    -------
    coeffs : (float, float)
        The slope (index 0) and intercept (index 1) for the line
        fit to the plot of log N (log of number of occupied boxes)
        vs log 1/l (log of inverse of box side length).

        The slope is the capacity dimension, which approximates the
        fractal dimension, and the intercept contains information
        about the total path length.

    residual : float
        The sum of residuals for the linear fit.
    """
    # Python method
    if debug or np.shape(points)[-1] != 2:
        return _capacityDimension_py(points, minL, maxL, sizeSteps, debug)

    # Numba compiled method
    return _capacityDimension_opt(points, minL, maxL, sizeSteps)

def _capacityDimension_py(points, minL=0.01, maxL=0.2, sizeSteps=30, debug=False):
    """
    Not intended to be called directly, but rather through
    the wrapper function `capacityDimension()`.
    
    Compute the capacity (Hausdorf-Besicovich) dimension
    of a set of points using the box-counting algorithm.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Points to calculate the correlation dimension of.

    minL : float
        The minimum value of the box size to check, given in
        units of the total size of the point cloud.
        
    maxL : float
        The maximum value of the box size to check, given in
        units of the total size of the point cloud.

    sizeSteps : int
        The number of box sizes to use in the provided range
        to compute the capacity dimension.

    debug : bool
        Whether to plot diagnostic information.

    Returns
    -------
    coeffs : (float, float)
        The slope (index 0) and intercept (index 1) for the line
        fit to the plot of log N (log of number of occupied boxes)
        vs log 1/l (log of inverse of box side length).

        The slope is the capacity dimension, which approximates the
        fractal dimension, and the intercept contains information
        about the total path length.

    residual : float
        The sum of residuals for the linear fit.
    """
    pointBounds = np.array([np.min(points, axis=0), np.max(points, axis=0)])
    dimRanges = pointBounds[1] - pointBounds[0]

    # Diagonal length of the cloud
    pointCloudSize = np.sqrt(np.sum(dimRanges**2))
    
    # Generate box sizes to check, which should be evenly
    # spaced on a log scale since we will event ually
    # be taking the log
    lArr = np.logspace(np.log10(minL*pointCloudSize), np.log10(maxL*pointCloudSize), sizeSteps)

    filledBoxesArr = np.zeros(sizeSteps)

    # Choose a few images to show if we have debug enabled
    # so the user can get an idea of what their box sizes
    # look like.
    debugImages = [0, sizeSteps//2, sizeSteps-1]
    if debug:
        debugIndex = 0
        fig, ax = plt.subplots(1, 3, figsize=((len(debugImages)*3, 5)))
        
    for i in range(sizeSteps):
        # Decide the lattice size for the given box size
        latticeSizeLargestDim = int(np.ceil(np.max(dimRanges) / lArr[i]))
        # Put all points between in the unit square between [0,0,...] and
        # [1,1,...]
        normPoints = (points - pointBounds[0]) / np.max(dimRanges)
        # Turn every point into the index of the box it belongs to
        normPoints *= latticeSizeLargestDim
        normPoints = np.floor(normPoints).astype(np.int64)

        # Now look at how many unique boxes are occupied
        uniqueBoxes = np.unique(normPoints, axis=0)

        if debug and i in debugImages:
            image = np.zeros(np.ceil(dimRanges / lArr[i]).astype(np.int64)+1)
            for ind in uniqueBoxes:
                image[tuple(ind)] = 1

            ax[debugIndex].imshow(image)
            debugIndex += 1
            
        filledBoxesArr[i] = len(uniqueBoxes)

    if debug:
        fig.tight_layout()
        plt.show()

    # Perform the linear fit
    res = np.polyfit(np.log(1/lArr), np.log(filledBoxesArr), 1, full=True)
    coeffs, residuals = res[0], res[1][0]
    
    if debug:
        plt.scatter(1/lArr, filledBoxesArr, c='tab:red')

        # We really only care about the slope here, so we just
        # start the line from the first datapoint, but this isn't
        # technically the whole fit information (it doesn't include the
        # intercept).
        plt.plot(1/lArr, filledBoxesArr[0]/lArr**coeffs[0], '--', label=f'Fit: $m = {coeffs[0]:.4}$')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$1/l$')
        plt.ylabel('$N$')
        plt.legend()
        plt.title('Capacity Dimension')
        plt.show()

    return coeffs[0], coeffs[1], residuals

@numba.njit(cache=True)
def _capacityDimension_opt(points, minL=0.01, maxL=0.2, sizeSteps=30):
    """
    Not intended to be called directly, but rather through the wrapper
    function `capacityDimension()`.
    
    Compute the capacity (Hausdorf-Besicovich) dimension
    of a set of points using the box-counting algorithm.

    Optimized using numba so runs much faster than
    `_capacityDimension_py()`, but can't make any debug
    plots. Also uses an approximate method to identify
    unique indices of boxes, meaning it might be slightly
    approximate in certain cases; in my testing it seems
    accurate at least up to 5-6 decimal places for the
    final fit parameters, but it could have issues in
    certain cases.

    This approximate method only works in 2D so this 
    method likely will not work for higher dimensional data.
    
    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Points to calculate the correlation dimension of.

    minL : float
        The minimum value of the box size to check, given in
        units of the total size of the point cloud.
        
    maxL : float
        The maximum value of the box size to check, given in
        units of the total size of the point cloud.

    sizeSteps : int
        The number of box sizes to use in the provided range
        to compute the capacity dimension.

    Returns
    -------
    coeffs : (float, float)
        The slope (index 0) and intercept (index 1) for the line
        fit to the plot of log N (log of number of occupied boxes)
        vs log 1/l (log of inverse of box side length).

        The slope is the capacity dimension, which approximates the
        fractal dimension, and the intercept contains information
        about the total path length.

    residual : float
        The sum of residuals for the linear fit.
    """
    # Only argmin and argmax allow axis keyword argument in
    # numba, so we have to do this manually
    pointBounds = np.zeros((2, points.shape[-1]))
    for i in range(pointBounds.shape[-1]):
        pointBounds[:,i] = [np.min(points[:,i]), np.max(points[:,i])]
        
    dimRanges = pointBounds[1] - pointBounds[0]

    # Diagonal length of the cloud
    pointCloudSize = np.sqrt(np.sum(dimRanges**2))
    
    # Generate box sizes to check, which should be evenly
    # spaced on a log scale since we will event ually
    # be taking the log
    lArr = np.logspace(np.log10(minL*pointCloudSize), np.log10(maxL*pointCloudSize), sizeSteps)

    filledBoxesArr = np.zeros(sizeSteps)

    for i in range(sizeSteps):
        # Decide the lattice size for the given box size
        latticeSizeLargestDim = int(np.ceil(np.max(dimRanges) / lArr[i]))
        # Put all points between in the unit square between [0,0,...] and
        # [1,1,...]
        normPoints = (points - pointBounds[0]) / np.max(dimRanges)
        # Turn every point into the index of the box it belongs to
        normPoints *= latticeSizeLargestDim
        normPoints = np.floor(normPoints).astype(np.int64)

        # Now look at how many unique boxes are occupied.
        # numba doesn't support unique with the axis keyword
        # so we just use an approximate alternative by
        # using a very nonlinear, asymmetric function of
        # the indices.
        def f(x):
            # Adding fractional exponents here vastly improves
            # the accuracy of this approximation, but it also
            # makes the calculation much slower. But since we're
            # using numba, in the end we still see a pretty big
            # speedup.
            return x[0]**2.1 + x[1]**3 #+ np.sum(x**2 * (np.arange(len(x))+1))
            
        identifyingValues = np.array([f(n) for n in normPoints])
        uniqueBoxes = np.unique(identifyingValues)
        #print(len(uniqueBoxes))
        #uniqueBoxes = np.unique(normPoints, axis=0)
        #print(len(uniqueBoxes))

        filledBoxesArr[i] = len(uniqueBoxes)

    # Perform the linear fit
    # numba doesn't support polyfit, so we have to do it
    # manually using linear algebra
    x = np.log(1/lArr)
    y = np.log(filledBoxesArr)

    # Want to solve the equation y = (b0 b1) @ (x 1)
    X = np.ones((2, len(x)))
    X[0] = x
    
    a = np.linalg.inv(np.dot(X, X.T))
    c = np.dot(X, y)
    b = np.dot(a, c)

    # Compute the residuals
    residuals = np.sum((y - np.dot(b, X))**2)

    # This is exactly equivalent to:    
    #res = np.polyfit(np.log(1/lArr), np.log(filledBoxesArr), 1, full=True)
    #b, residuals = res[0], res[1][0]
    
    return b[0], b[1], residuals
