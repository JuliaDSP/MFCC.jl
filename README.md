# MFCC

[![Build Status](https://travis-ci.org/JuliaDSP/MFCC.jl.svg?branch=master)](https://travis-ci.org/JuliaDSP/MFCC.jl.svg?branch=master) 

A package to compute Mel Frequency Cepstral Coefficients.

The essential routine is re-coded from Dan Ellis's rastamat package, and parameters are named similarly.

Please note that the feature-vector array consists of a vertical stacking of row-vector features.  This is consistent with the sense of direction of, e.g., `Base.cov()`, but inconsistent with, e.g., `DSP.spectrogram()` or `Clustering.kmeans()`.

`mfcc()` has many parameters, but most of these are set to defaults that _should_ mimick HTK default parameter (not thoroughly tested).

## Feature extraction main routine

```julia
mfcc(x::Vector, sr=16000.0, defaults::Symbol; args...)
```
Extract MFCC features from the audio data in `x`, using parameter settings characterized by `defaults`
- `:rasta`: defaults according to Dan Ellis' Rastamat package
- `:htk`: defaults mimicking defaults of HTK (unverified)
- `:nbspeaker`: narrow-band speaker recognition
- `:wbspeaker`: wide-band speaker recognition

The actual routine for MFCC computation has many parameters, these are basically the same parameters as in Dan Ellis's rastamat package.

```julia
mfcc(x::Vector, sr=16000.0; wintime=0.025, steptime=0.01, numcep=13, lifterexp=-22, sumpower=false, preemph=0.97, dither=false, minfreq=0.0, maxfreq=sr/2, nbands=20, bwidth=1.0, dcttype=3, fbtype=:htkmel, usecmp=false, modelorder=0)
```

  This is the main routine computing MFCCs.  `x` should be a 1D vector of `FloatingPoint` samples of speech, sampled at a frequency of `sr`.  Every `steptime` seconds, a frame of duration `wintime` is analysed.  The log energy in a filterbank of `nbands` bins is computed, and a cepstral (discrete cosine transform) representaion is made, keeping only the first `numcep` coefficients (including log energy).  The result is a tuple of three values:

 - a matrix of `numcep` columns with for each speech frame a row of MFCC coefficients
 - the power spectrum computed with `DSP.spectrogram()` from which the MFCCs are computed
 - a dictionary containing information about the parameters used for extracting the features.


## Pre-set feature extraction applications

We have defined a couple of standard sets of parameters that should function well for particular applications in speech technology.  They are accessible through the higher level function `feacalc()`.  The top-level interface for calculating features is
```julia
feacalc(wavfile::AbstractString, application::Symbol; kwargs...)
```
This will compute speech features suitable for a specific `application`, which currently can be one of:
- `:nbspeaker`: narrowband (telephone speech) speaker recognition: 19 MFCCs + log energy, delta's, energy-based speech activity detection, feature warping (399 samples)
- `:wbspeaker`: wideband speaker recognition: same as above but with wideband MFCC extraction parameters
- `:language`: narrowband language recognition: Shifted Delta Cepstra, energy-based speech activity detection, feature warping (299 samples)
- `:diarization`: 13 MFCCs, utterance mean and variance normalization

The `kwargs...` parameters allow for various options in file format, feature augmentation, speech activity detection and MFCC parameter settings.  They trickle down to versions of `feacalc()` and `mfcc()` allow for more detailed specification of these parameters.

`feacalc()` returns a tuple of three structures:
- an `Array` of features, one row per frame
- a `Dict` with metadata about the speech (length, SAD selected frames, etc.)
- a `Dict` with the MFCC parameters used for feature extraction

### More fine-grained control of `feacalc()`

```julia
feacalc(wavfile::AbstractString; method=:wav, kwargs...)
```
This function reads an audio file from disk and represents the audio as an `Array`, and then runs the feature extraction.

The `method` parameter determines what method is used for reading in the audio file:
- `:wav`: use Julia's native [WAV](https://github.com/dancasimiro/WAV.jl) library to read RIFF/WAVE `.wav` files
- `:sox`: use external `sox` program for figuring out the audio file format and converting to native represantation
- `:sph`: use external `w_decode` program to deal with (compressed) NIST sphere files

```julia
feacalc(x::Array; chan=:mono, augtype=:ddelta, normtype=:warp, sadtype=:energy, dynrange::Real=30., nwarp::Int=399, sr::AbstractFloat=8000.0, source=":array", defaults=:nbspeaker, mfccargs...)
```
The `chan` parameter specifies for which channel in the audio file you want features.  Possible values are:
- `:mono`: average all channels
- `:a`, `:b`, ...: Use the first (left), second (right), ... channel
- `c::Int`: Use the `c`th channel

The `augtype` parameter specifies how the speech features are augmented.  This can be:
- `:none` for no additional features
- `:delta` for 1st order derivatives
- `:ddelta` for first and second order derivatives
- `:sdc` for replacement od the MFCCs with shifted delta cepstra with the default parameters `n, d, p, k = 7, 1, 3, 7`

The `normtype` parameter specifies how the features are normalized after extraction
- `:none` for no normalization
- `:warp` for short-time Gaussianization using `nwarp` frames, see `warp()` below
- `:mvn` for mean and variance normalization, see `znorm()` below

The `sad` parameter controls if Speech Activity Detection is carried out on the features, filtering out frames that do not contain speech
- `:none`: apply no SAD
- `:energy`: apply energy based SAD, omitting frames with an energy less than `dynrange` below the maximum energy of the file.

The various applications actually have somewhat different parameter settings for the basic MFCC feature extraction, see the `defaults` parameter of `mfcc()` below.

### Feature warping, or short-time Gaussianization (Jason Pelecanos)
```julia
warp(x::Matrix, w=399)
```

 This tansforms columns of `x` by short-time Gaussianization.  Each value in the middle of `w` rows is replaced with its normal deviate (the quantile function of the normal distribution) based on its rank within the `w` values.  The result has the same dimensions as `x`, but the values are chosen from a discrete set of `w` normal deviates.

```julia
znorm(x::Matrix)
znorm!(x::Matrix)
```

This normalizes the data `x` on a column-by-column basis by an affine transformation, making the per-column mean 0 and variance 1.

### Short-term mean and variance normalization

As an alternative to short time Gaussianization, and similar to `znorm()`, you can compute the `znorm` for a sample in the centre of a sliding window of width `w`, where mean and variance are computed just over that window using
```julia
stmvn(x::Matrix, w=399)
```

### Derivatives

Derivative of features, fitted over `width` consecutive frames:
```julia
deltas(x::Matrix, width::Int)
```
The derivatives are computed over columns individually, and before the derivatives are computed the data is padded with repeats of the first and last rows.  The resulting matrix has the same size as `x`.  `deltas()` can be applied multiple times in order to get higher order derivatives.

### Shifted-Delta-Cepstra

SDCs are features used for spoken language recognition, derived from typically MFCCs

```julia
sdc(x::Matrix, n::Int=7, d::Int=1, p::Int=3, k::Int=7)
```

This function expands (MFCC) features in `x` by computing derivatives over `2d+1` consecutive frames for the first `n` columns of `x`, stacking derivatives shifted over `p` frames `k` times.  Before the calculation, zero adding is added so that the number of rows of the resuls is the same as for `x`.
