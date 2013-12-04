MFCCs
=====

This module allows computing speech type features, using the Rasta and SignalProcessing. 

The essential routine is re-coded from Dan Ellis's rastamat package.  

Please note that the feature-vector array `x` consists of a vertical stacking of row-vector features. 

Note that `mfcc()` has many parameters, but most of these are set to defaults that _should_ mimick HTK default parameter (untested). 

Feature extraction
 - `mfcc(x::Vector, sr::Number=16000.0; wintime=0.025, steptime=0.01, numce
p=13, lifterexp=-22, 
              sumpower=false, preemph=0.97, dither=false, minfreq=0.0, maxfreq=s
r/2, 
              nbands=20, bwidth=1.0, dcttype=3, fbtype=:htkmel, usecmp=false, m
odelorder=0)`

Feature warping, or short-time Gaussianization (Jason Pelecanos)
 - `warp(x::Array, w=399)`

Derivative of features, fitted over `width` consecutive frames:
 - deltas(x::Array, width::Int)

Shifted-Delta-Cepstra (features for spoken language recogntion)
 - `sdc(x::Array, n::Int=7, d::Int=1, p::Int=3, k::Int=7)`

