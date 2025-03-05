import numpy as np
import matplotlib.pyplot as plt

def renormalizeTrajectory(walk, l=None):
    """
    Resample a trajectory to have (roughly) evenly spaced steps,

    This step size can either scale with the minimum step size of the
    set of points, or be defined as an absolute value `l`.

    Can take a long time to compute if the trajectory is quite long
    and/or the provided step size is very small.
    
    Parameters
    ----------
    walk : numpy.ndarray[N,d]
        A sequence of points comprising the trajectory to
        be renormalized.

    l : float, optional
        The length scale that will be the (roughly) uniform step
        size after renormalization.

        If not provided, the size of the smallest step in the
        walk will be used.

    Returns
    -------
    newWalk : numpy.ndarray[M,d]
        The resampled trajectory.
    """
    # Compute the smallest step size
    directionVectors =  walk[1:] - walk[:-1]
    stepSizes = np.sqrt(np.sum(directionVectors**2, axis=-1))
    
    #plt.hist(stepSizes, bins=np.logspace(-7, 0, 20))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.show()
    
    if l is None:
        minStepSize = np.min(stepSizes)
    else:
        minStepSize = l
    
    # Find how many times we need to subdivide each step
    stepSubdivisions = np.ceil(stepSizes / minStepSize).astype(np.int32)

    # Interpolate each step as necessary
    totalSteps = np.sum(stepSubdivisions)

    newPoints = np.zeros((0, np.shape(walk)[-1]))
    
    for i in range(len(walk)-1):
        tArr = np.linspace(0, 1, stepSubdivisions[i])
        p = tArr[:,None]*directionVectors[i] + walk[i]
        newPoints = np.concatenate((newPoints, p))
        
    return newPoints


def rulerDimension(walk, minSkip=1, maxSkip=100, steps=50, meanStepScaling=None, average=True, debug=False):
    """
    Approximate the fractal dimension of a trajectory by measuring the
    length of trajectory as a function of the ruler length.

    We implement various "ruler lengths" by skipping $i$ points for many
    (integer) values of $i$. We then plot $\log{L_i}$ vs. $\log{i}$, and
    perform a linear fit.

    Note that in comparison to the other fractal dimension approximation
    methods (eg. capacity dimension, correlation dimension) this method
    requires that your data be a sequential set of points.

    This method is very similar to the original definition of fractal
    dimension by Mandlebrot [1], and the algorithm is similar to the
    Higuchi dimension calculation [2]. See `average` keyword argument
    for information about Higuchi method.

    Parameters
    ----------
    walk : numpy.ndarray[N,d]
        The points visited by the walk. Sequentual ordering is important!

    minSkip : int
        The smallest value of the skip factor.

    maxSkip : int
        The largest value of the skip factor.

    steps : int
        The number of skip values to check in the specified
        range.

    meanStepScaling : float, optional
        The new (uniform) step size when renormalizing the
        trajectory, given as a fraction of the current
        average step size of the walk.

        If not provided, no rescaling will be done. A value of
        zero will also mean that no rescaling is done, added for
        compatability with hdf5 files.

    average : bool, optional
        Naively, we can compute the length of a single downsampled
        trajectory sourced from X(i), eg. [X(1), X(4), X(7), ...].

        Higuchi [2] suggested that if you are downsampling by a factor
        $k$, then you actually have k different trajectories to measure
        the length of, eg. [X(1), X(4), X(7), ...], [X(2), X(5), X(8), ...],
        etc. 

        If `average=True`, all $k$ trajectories will be averaged for each
        length measurement. This means we have to remove the last $k$
        points from the trajectory, but usually $ k << N $ so this isn't
        a problem.

        Note that averaging does increase the computation time by roughly
        10x, though it reduces the typical error of the calculation as
        well (it is difficult to quantify exactly how much, though
        certainly the linear fits tend to be better using the averaging
        method).

    debug : bool, optional
        Whether to plot debug information.

    Returns
    -------
    D : float
        1 + slope for the linear fit to the plot of log L
        (log of path length) vs log i (log of skip factor).
        
        This value approximates the fractal dimension.
        
    intercept : float
        The intercept for the linear fit to the plot of log L
        (log of path length) vs log i (log of skip factor).

        This intercept contains information about the total path length.

    residual : float
        The sum of residuals for the linear fit.

    References
    ----------

    [1] Mandelbrot, B. (1967). How Long Is the Coast of Britain?
    Statistical Self-Similarity and Fractional Dimension. Science,
    156(3775), 636–638. https://doi.org/10.1126/science.156.3775.636

    [2] Higuchi, T. (1988). Approach to an irregular time series on the
    basis of the fractal theory. Physica D: Nonlinear Phenomena, 31(2),
    277–283. https://doi.org/10.1016/0167-2789(88)90081-4


    """
    # First, we need to normalize the step size for our walk.
    # It's not a bad idea set the renormalized length scale
    # below the mean step size, so we calculate that and then
    # just use a fraction of it.
    directionVectors =  walk[1:] - walk[:-1]
    stepSizes = np.sqrt(np.sum(directionVectors**2, axis=-1))

    if meanStepScaling is not None and meanStepScaling > 0:
        renormStepSize = np.mean(stepSizes)*meanStepScaling
    
        # Renormalize
        renormWalk = renormalizeTrajectory(walk, l=renormStepSize)
    else:
        renormWalk = walk
        
    skipArr = np.unique(np.logspace(np.log10(minSkip), np.log10(maxSkip), steps).astype(np.int32))
    lengthArr = np.zeros(len(skipArr))

    # Choose a few images to show if we have debug enabled
    # so the user can get an idea of what their box sizes
    # look like.
    if debug:
        debugImages = [0, len(skipArr)//2, len(skipArr)-1]
        debugIndex = 0
        fig, ax = plt.subplots(1, len(debugImages)+1, figsize=(((1+len(debugImages))*4, 4)))
    
    # If desired, we can average over multiple skipping trajectories, eg.
    # [X(1), X(4), X(7), ...] AND [X(2), X(5), X(8), ...] etc.
    # This was first proposed (to my knowledge) by Higuchi in 1988
    if average:
        for i in range(len(skipArr)):
            # Average over all k downsampled trajectories.
            for j in range(skipArr[i]):
                # Note the indexing for the beginning and end so that
                # we make sure each of the k trajectories has the same
                # length.
                dsPoints = renormWalk[j:-(skipArr[i]-j):skipArr[i]]
                lengthArr[i] += np.sum(np.sqrt(np.sum((dsPoints[1:] - dsPoints[:-1])**2, axis=-1))) / skipArr[i]

            # Debug plots
            if debug and i in debugImages:
                ax[debugIndex].set_title(f'Skip factor: {skipArr[i]}')
                ax[debugIndex].plot(*dsPoints.T)
                debugIndex += 1
    else:
        # Otherwise, we only use 1 of the k possible trajectories.
        for i in range(len(skipArr)):
            dsPoints = renormWalk[::skipArr[i]]
            lengthArr[i] = np.sum(np.sqrt(np.sum((dsPoints[1:] - dsPoints[:-1])**2, axis=-1)))

            # Debug plots
            if debug and i in debugImages:
                ax[debugIndex].set_title(f'Skip factor: {skipArr[i]}')
                ax[debugIndex].plot(*dsPoints.T)
                debugIndex += 1
        
    res = np.polyfit(np.log(1/skipArr), np.log(lengthArr), 1, full=True)
    coeffs, residuals = res[0], res[1][0]

    if debug:
        ax[-1].scatter(skipArr, lengthArr)
        ax[-1].plot(skipArr, lengthArr[0]*skipArr**(-coeffs[0]), '--', c='tab:red', label=f'Fit: $m={coeffs[0]:.3}$')
        ax[-1].set_title('Ruler Dimension')
        ax[-1].set_xscale('log')
        ax[-1].set_yscale('log')
        ax[-1].set_xlabel('Skip factor, $i$')
        ax[-1].set_ylabel('Total path length, $L_i$')
        ax[-1].legend()
        
        fig.tight_layout()
        plt.show()

    # For the ruler (or "divider") method, the actual
    # approximate fractal dimension is 1 + eta where
    # eta is the exponent we got from our fit.
    return 1 + coeffs[0], coeffs[1], residuals
