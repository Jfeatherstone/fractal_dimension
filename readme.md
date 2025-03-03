## Fractal Dimension

This library implements several methods for calculating the fractal
dimension of a time-series or collection of points.

Note that most techniques to compute fractal dimensions are for images;
this library does not implement those techniques (though of course
there is some overlap). The input data for these techniques should be
a collection of points, either ordered or unordered.

### Techniques

Many techniques have been described under different names by different
authors; I've done my best to list the common names for each technique
below.

#### Ruler/Divider/Higuchi Dimension

Assumes ordered points: Yes
Algorithm: Higuchi

Compute the length of the trajectory as a function of the smallest 
distance you are able to measure ("the size of your ruler").

In practice, we actually just downsample our trajectory by some integer
factor, and compute the length of the full trajectory.

#### Capacity/Box-counting Dimension

Assumes ordered points: No
Algorithm: Box-counting

Compute how many boxes of fixed size are required to encapsulate the
entire set of points, as a function of this box size.

#### Correlation Dimension

Assumes ordered points: No
Algorithm: Grassberger-Procaccia

Compute how many neighbors each point has as a function of distance
within which a point is considered a neighbor.

#### FFT Dimension

Assumes ordered points: Yes
Algorithm: Fast Fourier Transform

Compute the power law tail of the power spectrum as a function of
frequency.
