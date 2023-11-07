## runtests.jl  Unit tests for MFCC
## (c) 2015 David A. van Leeuwen
##
## Licensed under the MIT software license, see LICENSE.md

using WAV
using MFCC
using Test

import MFCC: sad, levinson, toeplitz

# test io

x, meta, params = feacalc("bl2.wav", normtype=:none, method=:wav, augtype=:none, sadtype=:none)
y = feaload("bl2.mfcc")

@assert x == y

t_fn = "mfccs/bl2.hdf5"

feasave(t_fn, x; meta=meta, params=params)
mfcc_tup = feaload(t_fn; meta=true, params=true)

@assert feasize(t_fn) == size(y)
@assert mfcc_tup == (x, meta, params)

rm(t_fn)

y, sr = wavread("bl2.wav"); y_mat = y
y, sr = vec(y), Float64(sr)

# test feacalc with different parameters
feacalc(y; normtype=:mvn)
feacalc(y_mat; sr=sr, chan=1)
feacalc(y_mat; sr=sr, chan=:a, augtype=:sdc)
feacalc(y; sr=sr, usecmp=true, modelorder=20, dcttype=1)

# test mfcc with htk, matrix input
mfcc(repeat(y, 1, 2), sr, :htk)

for defaults in (:nbspeaker, :wbspeaker, :language, :diarization)
    feacalc("bl2.wav", defaults)
end

for (fb, dcttype) in zip((:bark, :mel, :htkmel), (1, 3, 4))
    mfcc(y; usecmp=true, fbtype=fb, dcttype=dcttype)
end

speech = sad("bl2.wav", devnull, devnull)

z = warp(x)
z = warp(x, 100)
z = deltas(x)
z = deltas(x, 1)
z = znorm(x)
z = stmvn(x)

x = randn(100000)
l = levinson(x, 101)
z = lifter(x, 0.6, true)
z = stmvn(x, 400000)
z = warp(randn(1000))
p = powspec(x)
a = audspec(p)
a = postaud(p, 8000, :bark, true)

# test for invalid/unsupported arguments
@test_throws DomainError feacalc(y_mat; sr=sr, chan=2)
@test_throws ArgumentError feacalc(y_mat; sr=sr, chan=nothing)
@test_throws ArgumentError feacalc(y; sr=sr, usecmp=true, modelorder=1, dcttype=1)
@test_throws ArgumentError feacalc(y; sr=sr, usecmp=true, modelorder=1, dcttype=2)
@test_throws ArgumentError feacalc("bl2.wav", :bosespeaker)
@test_throws ArgumentError mfcc(y, sr, :pasta)
@test_throws ArgumentError postaud(a, 4000, :cough)
@test_throws "Lift number is too high (>10)" lifter(y, 100)
@test_throws "Negative lift must be integer" lifter(y, -0.6)
@test_throws ArgumentError levinson(Int[], 1)
@test_throws DomainError levinson(x, -1)

@test_warn "First elements of a Toeplitz matrix should be equal." toeplitz([1+im])

println("Tests passed")
