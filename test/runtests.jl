## runtests.jl  Unit tests for MFCC
## (c) 2015 David A. van Leeuwen
##
## Licensed under the MIT software license, see LICENSE.md

using WAV
using MFCC
using Test

import MFCC: sad, levinson, toeplitz

x, meta, params = feacalc("bl2.wav", normtype=:none, method=:wav, augtype=:none, sadtype=:none)

# test io
@testset "io" begin
    y = @test_nowarn feaload("bl2.mfcc")

    @test x == y

    t_fn = "../bl2.hdf5"

    @test_nowarn feasave(t_fn, x; meta=meta, params=params)
    mfcc_tup = @test_nowarn feaload(t_fn; meta=true, params=true)

    @test feasize(t_fn) == size(y)
    @test mfcc_tup == (x, meta, params)

    rm(t_fn)
end

y, sr = wavread("bl2.wav"); y_mat = y
y, sr = vec(y), Float64(sr)

# test feacalc with different parameters
@testset "feacalc" begin
    @test_nowarn feacalc(y; normtype=:mvn)
    @test_nowarn feacalc(y_mat; sr=sr, chan=1)
    @test_nowarn feacalc(y_mat; sr=sr, chan=:a, augtype=:sdc)
    @test_nowarn feacalc(y; sr=sr, usecmp=true, modelorder=20, dcttype=1)
    for defaults in (:nbspeaker, :wbspeaker, :language, :diarization)
        @test_nowarn feacalc("bl2.wav", defaults)
    end
end

@testset "mfcc" begin
    # test mfcc with htk, matrix input
    @test_nowarn mfcc(repeat(y, 1, 2), sr, :htk)

    for (fb, dcttype) in zip((:bark, :mel, :htkmel), (1, 3, 4))
        @test_nowarn mfcc(y; usecmp=true, fbtype=fb, dcttype=dcttype)
    end
end

@testset "rasta + utilities" begin
    @test_nowarn sad("bl2.wav", devnull, devnull)

    @test_nowarn warp(x)
    @test_nowarn warp(x, 100)
    @test_nowarn deltas(x)
    @test_nowarn deltas(x, 1)
    @test_nowarn znorm(x)
    @test_nowarn stmvn(x)

    v = randn(100000)
    @test_nowarn levinson(v, 101)
    @test_nowarn lifter(v, 0.6, true)
    @test_nowarn stmvn(v, 400000)
    @test_nowarn warp(randn(1000))
    p = @test_nowarn powspec(v)
    a = @test_nowarn audspec(p)
    a = @test_nowarn postaud(a, 8000, :bark, true)
end

# test for invalid/unsupported arguments
@testset "Test throws for invalid arguments" begin
    @test_throws DomainError feacalc(y_mat; sr=sr, chan=2)
    @test_throws ArgumentError feacalc(y_mat; sr=sr, chan=nothing)
    @test_throws ArgumentError feacalc(y; sr=sr, usecmp=true, modelorder=1, dcttype=1)
    @test_throws ArgumentError feacalc(y; sr=sr, usecmp=true, modelorder=1, dcttype=2)
    @test_throws ArgumentError feacalc("bl2.wav", :bosespeaker)
    @test_throws ArgumentError mfcc(y, sr, :pasta)
    @test_throws ArgumentError postaud(y_mat, 4000, :cough)
    @test_throws DomainError lifter(y, 100)
    @test_throws DomainError lifter(y, -0.6)
    @test_throws ArgumentError levinson(Int[], 1)
    @test_throws DomainError levinson(x, -1)

    @test_logs (:warn, "First elements of a Toeplitz matrix should be equal.") toeplitz([1 + im])
end

println("Tests passed")
