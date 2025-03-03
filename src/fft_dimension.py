import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq

def fftDimension(walk, debug=False):
    """
    Compute the power law tail of the power spectrum
    using Fast Fourier Transform.

    This exponent can approximate the fractal dimension
    of the system in certain cases.

    Parameters
    ----------
    walk : numpy.ndarray[N,d]
        The points visited by the walk. Sequentual ordering is important!

    debug : bool
        Whether to plot debug information.

    Returns
    -------
    D : float
        The magnitude of the slope for the linear fit to the
        log-log power spectrum.

        This value approximates the fractal dimension in some cases.

    intercept : float
        The intercept for the linear fit to the log-log power spectrum.

        This intercept contains information about the total path length.

    residual : float
        The sum of residuals for the linear fit.

    """
    directionVectors =  walk[1:] - walk[:-1]
    stepSizes = np.sqrt(np.sum(directionVectors**2, axis=-1))

    # If the length of the walk is even, we have to remove
    # the zero frequency value in the middle.
    avgPowerSpec = np.zeros(len(walk)//2 + len(walk)%2 - 1)

    # Average over each dimension separately
    # I'm not sure if this is actually how you are supposed to do
    # this for a multidimensional walk, but I can't find any other
    # better ideas. When I use just the array of step sizes, I
    # don't get a power law decay as expected. TODO
    for i in range(np.shape(walk)[-1]):
        powerSpectrum = fft(walk[:,i])**2
        # The frequency values will range from [-a, ..., 0, ..., a]
        # so we will split them apart in a second
        freqArr = fftfreq(len(powerSpectrum))

        # Now have to split into positive and negative to average
        posIndices = np.where(freqArr > 0)
        negIndices = np.where(freqArr < 0)

        posFreq = freqArr[posIndices]
        posFreqSpec = powerSpectrum[posIndices]

        # If the length of our walk is even, the frequency bins will
        # not be symmetric, so we have to remove 1 extra entry on the
        # negative side
        negFreq = freqArr[negIndices][None if len(powerSpectrum)%2 else 1:]
        negFreqSpec = powerSpectrum[negIndices][None if len(powerSpectrum)%2 else 1:]

        # Average the real components
        avgPowerSpec += np.real(negFreqSpec[::-1] + posFreqSpec)/(2*np.shape(walk)[-1])

    if debug:
        plt.scatter(posFreq, avgPowerSpec, s=5)
        plt.plot(posFreq, avgPowerSpec[-1]*posFreq**(-2), '--', c='tab:red')
        plt.yscale('log')
        plt.xscale('log')
        plt.show()

    # Note that the power spectrum usually looks quite spread out, so often
    # you don't get a super nice linear fit as you would in the other
    # methods of approximating fractal dimension. I'm not sure if there
    # is an agreed-upon technique for this, but that might be a good
    # thing to check. TODO
    res = np.polyfit(np.log(np.real(posFreq)), np.log(np.abs(avgPowerSpec)), 1, full=True)
    coeffs, residuals = res[0], res[1][0]

    return -coeffs[0], coeffs[1], residuals


def limitLinFit(x, y, topFrac=.35, bins=30):
    """
    Fit a straight line to the top and bottom edges
    of a set of points.

    Developed to be used for fitting the upper limit of
    a power spectrum, since they tend to be very noisy,
    but I'm not sure if this piece of information is actually
    helpful at all.
    """
    # Slight padding on the top to make sure the last point is included.
    xBins = np.linspace(np.min(x), np.max(x)*1.01, bins)

    # Split up the y values based on the x bins
    binnedY = [np.zeros(0)]*bins
    binnedX = [np.zeros(0)]*bins

    for i in range(bins-1):
        indices = np.where((x >= xBins[i]) & (x < xBins[i+1]))
        binnedY[i] = y[indices]
        binnedX[i] = x[indices]


    #print(binnedY)
    binRanges = [(np.max(by) - np.min(by)) if len(by) > 1 else np.nan for by in binnedY]
    rangeWidth = np.nanmean(binRanges) * topFrac

    topX = np.zeros(0)
    topY = np.zeros(0)
    botX = np.zeros(0)
    botY = np.zeros(0)

    for i in range(bins):
        if len(binnedY[i]) == 0:
            continue

        topX = np.concatenate((topX, binnedX[i][np.where(binnedY[i] >= np.nanmax(binnedY[i]) - rangeWidth)]))
        topY = np.concatenate((topY, binnedY[i][np.where(binnedY[i] >= np.nanmax(binnedY[i]) - rangeWidth)]))

        botX = np.concatenate((botX, binnedX[i][np.where(binnedY[i] <= np.nanmin(binnedY[i]) + rangeWidth)]))
        botY = np.concatenate((botY, binnedY[i][np.where(binnedY[i] <= np.nanmin(binnedY[i]) + rangeWidth)]))

    plt.scatter(x, y, c='tab:blue')
    plt.scatter(topX, topY, c='tab:red')
    plt.scatter(botX, botY, c='tab:orange')

    #for i in range(len(topPointsX)):
    #    plt.scatter(topPointsX[i], topPointsY[i], c='tab:red')

    #plt.yscale('log')
    #plt.xscale('log')
    plt.show()

    res = np.polyfit(topX, topY, 1, full=True)
    coeffs, residuals = res[0], res[1][0]
    print(coeffs, residuals)

    res = np.polyfit(botX, botY, 1, full=True)
    coeffs, residuals = res[0], res[1][0]
    print(coeffs, residuals)

    return coeffs[0], coeffs[1], residuals

