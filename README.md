MFCC
=====

This module computes Mel Frequency Cepstral Coeffiencts.

The essential routine is re-coded from Dan Ellis's rastamat package, and parameters are named similarly.   

Please note that the feature-vector array consists of a vertical stacking of row-vector features.  This is consistent with the sense of direction of, e.g., `Base.cov()`, but inconsistent with, e.g., `DSP.spectrogram()` or `Clustering.kmeans()`. 

`mfcc()` has many parameters, but most of these are set to defaults that _should_ mimick HTK default parameter (not thoroughly tested). 

Feature extraction main routine
-------------------------------

```julia
mfcc(x::Vector, sr::FloatingPoint=16000.0; wintime=0.025, steptime=0.01, 
     numcep=13, lifterexp=-22, sumpower=false, preemph=0.97, 
     dither=false, minfreq=0.0, maxfreq=sr/2, nbands=20, bwidth=1.0, 
     dcttype=3, fbtype=:htkmel, usecmp=false, modelorder=0)
```

  This is the main routine computing MFCCs.  `x` should be a 1D vector of `FloatingPoint` samples of speech, sampled at a frequency of `sr`.  Every `steptime` seconds, a frame of duration `wintime` is analysed.  The log energy in a filterbank of `nbands` bins is computed, and a cepstral (discrete cosine transform) representaion is made, keeping only the first `numcep` coefficients (including log energy).  The result is a tuple of three values:
 
 - a matrix of `numcep` columns with for each speech frame a row of MFCC coefficients
 - the power spectrum computed with `DSP.spectrogram()` from which the MFCCs are computed
 - a dictionary containing information about the parameters used for extracting the features. 


Feature warping, or short-time Gaussianization (Jason Pelecanos)
----------------------------------------------------------------

`warp(x::Matrix, w=399)`
 
 This tansforms columns of `x` by short-time Gaussianization.  Each value in the middle of `w` rows is replaced with its normal deviate (the quantile function of the normal distribution) based on its rank within the `w` values.  The result has the same dimensions as `x`, but the values are chosen from a discrete set of `w` normal deviates. 

```julia
znorm(x::Matrix)
znorm!(x::Matrix)
```
 
This normalizes the data `x` on a column-by-column basis by an affine transformation, making the per-column mean 0 and variance 1.

Derivatives
-----------

Derivative of features, fitted over `width` consecutive frames:
```julia
deltas(x::Matrix, width::Int)
```
The derivatives are computed over columns individually, and before the derivatives are computed the data is padded with repeats of the first and last rows.  The resulting matrix has the same size as `x`.  `deltas()` can be applied multiple times in order to get higher order derivatives. 

Shifted-Delta-Cepstra 
----------------------
SDCs are features used for spoken language recognition, derived from typically MFCCs

```
sdc(x::Matrix, n::Int=7, d::Int=1, p::Int=3, k::Int=7)
```

This function expands (MFCC) features in `x` by computing derivatives over `2d+1` consecutive frames for the first `n` colimns of `x`, stacking derivatives shifted over `p` frames `k` times.  Before the calculation, zero adding is added so that the number of rosws of the resuls is the same as for `x`.  
