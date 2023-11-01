## runtests.jl  Unit tests for MFCC
## (c) 2015 David A. van Leeuwen
##
## Licensed under the MIT software license, see LICENSE.md

using WAV
using MFCC
using Test

import MFCC.sad

# test io

x, meta, params = feacalc("bl2.wav", normtype=:none, method=:wav, augtype=:none, sadtype=:none)
y = feaload("bl2.mfcc")

@assert x == y

t_fn = "bl2.hdf5"

feasave(t_fn, x; meta=meta, params=params)
mfcc_tup = feaload(t_fn; meta=true, params=true)

@assert feasize(t_fn) == size(y)
@assert mfcc_tup == (x, meta, params)

rm(t_fn)

y, sr = wavread("bl2.wav"); sr = Float64(sr)

# test feacalc with different parameters
feacalc(vec(y); normtype=:mvn)
feacalc(y; sr=sr, chan=1)
feacalc(y; sr=sr, chan=:a, augtype=:sdc)
feacalc(y; sr=sr, usecmp=true, modelorder=20, dcttype=1)

# test mfcc with htk
mfcc(y, sr, :htk)

# test for invalid/unsupported arguments
@test_throws DomainError feacalc(y; sr=sr, chan=2)
@test_throws ArgumentError feacalc(y; sr=sr, usecmp=true, modelorder=1, dcttype=1)
@test_throws ArgumentError feacalc(y; sr=sr, usecmp=true, modelorder=1, dcttype=2)
@test_throws ArgumentError feacalc("bl2.wav", :bosespeaker)

for defaults in (:nbspeaker, :wbspeaker, :language, :diarization)
    feacalc("bl2.wav", defaults)
end

mfcc(y; fbtype=:bark)
mfcc(y; fbtype=:mel)
mfcc(y; fbtype=:fcmel)

speech = sad("bl2.wav", devnull, devnull)

z = warp(x)
z = warp(x, 100)
z = deltas(x)
z = znorm(x)
z = stmvn(x)

x = randn(100000)
z = stmvn(x, 400000)
p = powspec(x)
a = audspec(p)

println("Tests passed")
