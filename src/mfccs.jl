## mfccs.jl
## (c) 2013--2014 David A. van Leeuwen

## Recoded from / inspired by melfcc from Dan Ellis's rastamat package.

using SpecialFunctions
using Distributed

## This function accepts a vector of sample values, below we will generalize to arrays,
## i.e., multichannel data
## Recoded from rastamat's "melfcc.m" (c) Dan Ellis.
## Defaults here are HTK parameters, this is contrary to melfcc
function mfcc(x::Vector{T}, sr::Real=16000.0; wintime=0.025, steptime=0.01, numcep=13,
              lifterexp=-22, sumpower=false, preemph=0.97, dither=false, minfreq=0.0, maxfreq=sr/2,
              nbands=20, bwidth=1.0, dcttype=3, fbtype=:htkmel,
              usecmp=false, modelorder=0) where {T<:AbstractFloat}
    if (preemph != 0)
        x = filt(TFFilter([1., -preemph], [1.]), x)
    end
    pspec = powspec(x, sr, wintime=wintime, steptime=steptime, dither=dither)
    aspec = audspec(pspec, sr, nfilts=nbands, fbtype=fbtype, minfreq=minfreq, maxfreq=maxfreq, sumpower=sumpower, bwidth=bwidth)
    if usecmp
        #  PLP-like weighting/compression
        aspec = postaud(aspec, maxfreq, fbtype)
    end
    if modelorder > 0
        if dcttype != 1
            error("Sorry, modelorder>0 and dcttype ≠ 1 is not implemented")
        end
        # LPC analysis
        lpcas = dolpc(aspec, modelorder)
        # cepstra
        cepstra = lpc2cep(lpcas, numcep)
    else
        cepstra = spec2cep(aspec, numcep, dcttype)
    end
    cepstra = Array(lifter(cepstra, lifterexp)')
    meta = Dict("sr" => sr, "wintime" => wintime, "steptime" => steptime, "numcep" => numcep,
            "lifterexp" => lifterexp, "sumpower" => sumpower, "preemph" => preemph,
            "dither" => dither, "minfreq" => minfreq, "maxfreq" => maxfreq, "nbands" => nbands,
            "bwidth" => bwidth, "dcttype" => dcttype, "fbtype" => fbtype,
            "usecmp" => usecmp, "modelorder" => modelorder)
    return (cepstra, pspec', meta)
end

mfcc(x::Array{T}, sr::Real=16000.0; args...) where {T<:AbstractFloat} = @distributed (tuple) for i=1:size(x)[2] mfcc(x[:,i], sr; args...) end


## default feature configurations, :rasta, :htk, :spkid_toolkit, :wbspeaker
## With optional extra agrs... you can specify more options
function mfcc(x::Vector{T}, sr::AbstractFloat, defaults::Symbol; args...) where {T<:AbstractFloat}
    if defaults == :rasta
        mfcc(x, sr; lifterexp=0.6, sumpower=true, nbands=40, dcttype=2, fbtype=:mel, args...)
    elseif defaults ∈ [:spkid_toolkit, :nbspeaker]
        mfcc(x, sr; lifterexp=0.6, sumpower=true, nbands=30, dcttype=2, fbtype=:mel, minfreq=130., maxfreq=3900., numcep=20, args...)
    elseif defaults == :wbspeaker
        mfcc(x, sr; lifterexp=0.6, sumpower=true, nbands=63, dcttype=2, fbtype=:mel, minfreq=62.5, maxfreq=7937.5, numcep=20, args...)
    elseif defaults == :htk
        mfcc(x, sr; args...)
    else
        error("Unknown set of defaults ", defaults)
    end
end

## our features run down with time, this is essential for the use of DSP.filt()
function deltas(x::Matrix{T}, w::Int=9) where {T<:AbstractFloat}
    (nr, nc) = size(x)
    if nr==0 || w <= 1
        return x
    end
    hlen = floor(Int, w/2)
    w = 2hlen+1                 # make w odd
    win = collect(convert(T, hlen):-1:-hlen)
    x1 = reshape(x[1,:], 1, size(x,2)) ## julia v0.4 v0.5 changeover
    xend = reshape(x[end,:], 1, size(x,2))
    xx = vcat(repeat(x1, hlen, 1), x, repeat(xend, hlen, 1)) ## take care of boundaries
    norm = 3 / (hlen * w * (hlen+1))
    return norm * filt(TFFilter(win, [1.]), xx)[2hlen.+(1:nr),:]
end


import Base.Sort.sortperm
sortperm(a::Array, dim::Int) = mapslices(sortperm, a, dims=dim)

function warpstats(x::Matrix{T}, w::Int=399) where {T<:Real}
    nx, nfea = size(x)
    wl = min(w, nx)
    hw = (wl+1) / 2
    erfinvtab = √2 * erfinv.(collect(1:wl) / hw .- 1)
    rank = similar(x, Int)
    if nx < w
        index = sortperm(x, 1)
        for j in 1:nfea
            rank[index[:,j], j] = collect(1:nx)
        end
    else
        for j in 1:nfea            # operations in outer loop over columns, better for memory cache
            for i in 1:nx
                s = max(1, i - round(Int, hw) + 1)
                e = s + w - 1
                if e > nx
                    d = e - nx
                    e -= d
                    s -= d
                end
                r = 1
                for k in s:e
                    r += x[i, j] > x[k, j]
                end
                rank[i, j] = r
            end
        end
    end
    return rank, erfinvtab
end

function warp(x::Matrix{T}, w::Int=399) where {T<:Real}
    rank, erfinvtab = warpstats(x, w)
    return erfinvtab[rank]
end
warp(x::Vector{T}, w::Int=399) where {T<:Real} = warp(x'', w)

function WarpedArray(x::Matrix, w::Int)
    rank, erfinvtab = warpstats(x, w)
    WarpedArray(rank, erfinvtab)
end

## mean and variance normalization
znorm(x::Array, dim::Int=1) = znorm!(copy(x), dim)
znorm!(x::Array, dim::Int=1) = broadcast!(/, x, broadcast!(-, x, x, mean(x, dims=dim)), std(x, dims=dim))

## short-term mean and variance normalization
function stmvn(x::Vector, w::Int=399)
    y = similar(x)
    hw = w ÷ 2 ## effectively makes `w` odd...
    nx = length(x)
    nx ≤ 1 && return x
    ## initialize sum x and sum x^2
    sx = (hw+1) * x[1]
    sxx = (hw+1) * x[1] * x[1]
    for i = 2:min(nx, hw+1)
        sx += x[i]
        sxx += x[i] * x[i]
    end
    if hw+1 > nx
        sx += (hw+1 - nx) * x[nx]
        sxx += (hw+1 - nx) * x[nx] * x[nx]
    end
    for i = 1:nx
        μ = sx / w
        σ = sqrt((sxx - μ * sx) / (w - 1))
        y[i] = (x[i] - μ) / σ
        mi = max(i - hw, 1)
        ma = min(i + hw + 1, nx)
        sx += x[ma] - x[mi]
        sxx += x[ma] * x[ma] - x[mi] * x[mi]
    end
    return y
end

stmvn(m::Matrix, w::Int=399, dim::Int=1) = mapslices(x->stmvn(x, w), m, dims=dim)

## Shifted Delta Coefficients by copying
function sdc(x::Matrix{T}, n::Int=7, d::Int=1, p::Int=3, k::Int=7) where {T<:AbstractFloat}
    nx, nfea = size(x)
    n <= nfea || error("Not enough features for n")
    hnfill = (k-1) * p / 2
    nfill1, nfill2 = floor(Int, hnfill), ceil(Int, hnfill)
    xx = vcat(zeros(eltype(x), nfill1, n), deltas(x[:,1:n], 2d+1), zeros(eltype(x), nfill2, n))
    y = zeros(eltype(x), nx, n*k)
    for (dt,offset) = zip(0:p:p*k-1, 0:n:n*k-1)
        y[:,offset+(1:n)] = xx[dt+(1:nx),:]
    end
    return y
end
