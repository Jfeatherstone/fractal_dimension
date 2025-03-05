import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

def correlationDimension(points, minR=5., maxR=30., sizeSteps=30, debug=False):
    """
    Compute the correlation dimension of a set of points
    using the Grassberger-Procaccia algorithm.

    Scales as the square of the number of points since we have
    to check in the neighborhood of each point, though using
    a KD Tree structure cuts down the computation time significantly.

    There are multiple ways you can approach this calculation; the current
    one is optimized to use as little memory as possible, though sacrifices
    some computational efficiency to do this. See the commented out code
    for an alternative method that likely evaluates quicker, but uses
    more memory.

    You have to be careful that the calculation isn't biased if
    the majority of points that are found nearby are sequential,
    ie. we are just measuring artifacts because our sampling frequency
    is too high. As such, it is not recommended to use a `minR` value
    of less than `1`.

    I briefly tried to optimize this using numba but because of the
    use of a KDTree, I couldn't get much improvement.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Points to calculate the correlation dimension of.

    minR : float
        The minimum value of the radius to check, given in
        units of the mean nearest neighbor distance.

    maxR : float
        The maximum value of the radius to check, given in
        units of the mean nearest neighbor distance.

    sizeSteps : int
        The number of sizes to sample between `minR` and `maxR`
        (spaced evenly on a logarithmic scale).

    debug : bool
        Whether to plot debug information.
    """
     # Construct a KD-Tree
    kdTree = KDTree(points)

    # Compute the mean neighbor distance
    nnDistances, nnIndices = kdTree.query(points, 2)
    avgNNDistance = np.mean(nnDistances[:,1])

    # Generate sphere sizes to check, which should be evenly
    # spaced on a log scale since we will eventually
    # be taking the log
    # And we square it since we will compare to values of r**2
    rArr = np.logspace(np.log10(minR*avgNNDistance), np.log10(maxR*avgNNDistance), sizeSteps)
    rSqrArr = rArr**2

    avgNeighborsArr = np.zeros(sizeSteps)
    # Note for the current method, we actually can't
    # calculate the variance; this is only possible
    # using the more memory-intensive method that is
    # currently commented out.
    varNeighborsArr = np.zeros(sizeSteps)

    # It might be slightly faster to use query_ball_tree from
    # the KDTree, but this means you need a lot more memory to
    # store that info for every point at once. I've chosen this
    # method of doing it one by one such that the memory usage is
    # minimal
    for i in range(len(points)):
        # Find neighbors using the largest r value, since that means
        # we can just use that subset for every other r value.
        neighbors = kdTree.query_ball_point(points[i], rArr[-1])

        # Compute distance from every point to it's neighbors
        distances = np.sum((points[neighbors] - points[i])**2, axis=-1)
        for j in range(len(rSqrArr)):
             # -1 is to account for the point counting itself as a neighbor
             avgNeighborsArr[j] += len(np.where(distances <= rSqrArr[j])[0]) - 1

    avgNeighborsArr /= len(points)

    # This is the more memory intensive method (sometimes needs up
    # to ~10GB), though it is often faster.
    # Note that we can't easily compute the variance
    # in the above method, but we can for this method.

    # # Precompute all neighbors and distances
    # neighbors = kdTree.query_ball_tree(kdTree, rArr[-1])
    # distances = [np.sum((points[neighbors[i]] - points[i])**2, axis=-1) for i in range(len(points))]

    # for i in range(len(rSqrArr)):
    #     numNeighbors = np.zeros(len(points))
    #     for j in range(len(points)):
    #         # -1 is to account for the point counting itself as a neighbor
    #         numNeighbors[j] = len(np.where(distances[j] <= rSqrArr[i])[0]) - 1

    #     avgNeighborsArr[i] = np.mean(numNeighbors)
    #     varNeighborsArr[i] = np.var(numNeighbors)

    res = np.polyfit(np.log(1/rArr), np.log(avgNeighborsArr), 1, full=True)
    coeffs, residuals = res[0], res[1][0]

    if debug:
        plt.scatter(1/rArr, avgNeighborsArr, c='tab:red')
        plt.fill_between(1/rArr,
                         avgNeighborsArr - np.sqrt(varNeighborsArr),
                         avgNeighborsArr + np.sqrt(varNeighborsArr),
                         alpha=0.1, color='tab:red')

        # Not actually plotting the proper fit here, just showing the
        # slope, since that's the most important part to visualize.
        plt.plot(1/rArr, avgNeighborsArr[0]/rArr**coeffs[0], '--', label=f'Fit: $m = {-coeffs[0]:.4}$')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$1/r$')
        plt.ylabel('Number of neighbors, $N$')
        plt.title('Correlation Dimension')
        plt.legend()
        plt.show()

    return -coeffs[0], coeffs[1], residuals

