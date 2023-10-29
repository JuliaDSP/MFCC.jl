## runtests.jl  Unit tests for MFCC
## (c) 2015 David A. van Leeuwen
##
## Licensed under the MIT software license, see LICENSE.md

using WAV
using MFCC
using SpecialFunctions
using Statistics

x, meta, params = feacalc("bl2.wav", normtype=:none, method=:wav, augtype=:none, sadtype=:none)
y = feaload("bl2.mfcc")

@assert x == y

t_fn = "bl2.hdf5"

feasave(t_fn, x; meta=meta, params=params)
mfcc_tup = feaload(t_fn; meta=true, params=true)

@assert feasize(t_fn) == size(y)
@assert mfcc_tup == (x, meta, params)

rm(t_fn)

feacalc(x)
feacalc("bl2.wav")
y = wavread("bl2.wav")[1]
mfcc(y; fbtype=:bark)
mfcc(y; fbtype=:mel)
mfcc(y; fbtype=:fcmel)

z = warp(x)
z = deltas(x)
z = znorm(x)
z = stmvn(x)

x = randn(100000)
p = powspec(x)
a = audspec(p)

println("Tests passed")
