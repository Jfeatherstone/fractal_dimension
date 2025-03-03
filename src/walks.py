"""
Implementations of random walks
"""
import numpy as np
import numba


@numba.njit()
def gaussianDistribution(mu=0, s=1, size=None):
    """
    """
    return np.abs(np.random.normal(mu, s, size=size))

@numba.njit()
def exponentialDistribution(a=1, b=1, size=None):
    """
    Returns random value from an unsigned exponential distribution
    """
    return -a*np.log(np.random.uniform(1e-10, 1, size=size)*b)

@numba.njit()
def powerLawDistribution(alpha=1, a=1, size=None):
    """
    Returns random value from an unsigned power law distribution
    """
    return (np.random.uniform(1e-10, 1e5, size=size)/a)**(1/(-1 - alpha))


@numba.njit()
def randomWalkIter(currPos, stepSize):
    """
    Iterative step for a random walk, optimized using numba.
    """
    d = len(currPos)
    # One angle ranges from [0, 2pi], all others from [0, pi]
    angleArr = [np.random.uniform(0, np.pi) for _ in range(d-2)] + [np.random.uniform(0, 2*np.pi)]

    # Choose a step size from a random distribution
    r = stepSize

    if d == 1:
        newPos = r + currPos
    else:
        newPos = np.zeros_like(currPos)
        # x_i = r cos(\theta_i) \prod_j^{i-1} sin(\theta_j)
        for j in range(d-1):
            newPos[j] = r * np.cos(angleArr[j])
            for k in range(j):
                newPos[j] *= np.sin(angleArr[k])
        #newPos[:-1] = [r * np.product(np.sin(angleArr[:j])) * np.cos(angleArr[j]) for j in range(d-1)]
        # x_n = r \prod_j^{i} sin(\theta_j)
        newPos[-1] = newPos[-2] * np.tan(angleArr[-1])
        newPos += currPos

    return newPos


STEPSIZE_DISTRIBUTIONS = {"gaussian": gaussianDistribution,
                          "exponential": exponentialDistribution,
                          "power": powerLawDistribution}

@numba.njit()
def stochasticWalk(steps, d=2, stepDist='gaussian', mu=0., s=1., a=1., b=1., alpha=1.):
    """
    Perform a stochastic walk with a prescribed step size
    distribution in arbitrary dimension.
    """

    # First, have to generate all of the step sizes
    if stepDist == 'gaussian':
        stepSizeArr = gaussianDistribution(mu, s, size=steps)
    elif stepDist == 'exponential':
        stepSizeArr = exponentialDistribution(a, b, size=steps)
    elif stepDist == 'power':
        stepSizeArr = powerLawDistribution(alpha, a, size=steps)

    # 1D is very easy, we just have to choose a direction left or right
    if d == 1:
        return np.cumsum(stepSizeArr * np.random.choice(np.array([1., -1.]), size=steps))[:,None]

    # Next, we generate the directions of each step in generalized
    # spherical coordinates.
    # One angle ranges from [0, 2pi], all others from [0, pi]
    angleArr = np.zeros((steps, d-1))
    for i in range(d-2):
        angleArr[:,i] = np.random.uniform(0, np.pi, size=steps)

    # The unique angle 
    angleArr[:,-1] = np.random.uniform(0, 2*np.pi, size=steps)

    # Compute sin and cos since we will need them
    sinAngleArr = np.sin(angleArr)
    cosAngleArr = np.cos(angleArr)
  
    # We actually need the cumulative product of these
    for i in range(steps):
        sinAngleArr[i] = np.cumprod(sinAngleArr[i])

    # This is the step in each direction we take at each time step
    stepArr = np.zeros((steps, d))

    for i in range(d-1):
        stepArr[:,i] = stepSizeArr * cosAngleArr[:,i] * sinAngleArr[:,i]


    # One dimension is very easy to calculate, it is just the product of
    # the sin of every angle times the step size
    stepArr[:,-1] = stepSizeArr * sinAngleArr[:,-1]

    # Now calculate the cumulative sum to get the absolute position
    # over time
    walkArr = np.zeros_like(stepArr)
    for i in range(d):
        walkArr[:,i] = np.cumsum(stepArr[:,i])

    return walkArr

# Much simpler alternative
# def randomWalk(steps, d=2):
#     return np.cumsum(np.random.uniform(-1, 1, size=(steps, d)), axis=0)
