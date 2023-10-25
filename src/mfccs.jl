## mfccs.jl
## (c) 2013--2014 David A. van Leeuwen

## Recoded from / inspired by melfcc from Dan Ellis's rastamat package.

using SpecialFunctions: erfinv
using Distributed
using DSP

## This function accepts a vector of sample values, below we will generalize to arrays,
## i.e., multichannel data
## Recoded from rastamat's "melfcc.m" (c) Dan Ellis.
## Defaults here are HTK parameters, this is contrary to melfcc
function mfcc(x::Vector{T}, sr::Real=16000.0; wintime=0.025, steptime=0.01, numcep=13,
              lifterexp=-22, preemph=0.97, minfreq=0.0, maxfreq=sr/2, nbands=20,
              bwidth=1.0, dcttype=3, fbtype=:htkmel, modelorder=0, sumpower::Bool=false,
              dither::Bool=false, usecmp::Bool=false) where {T<:AbstractFloat}
    if !iszero(preemph)
        x = filt(PolynomialRatio([1., -preemph], [1.]), x)
    end
    pspec = powspec(x, sr; wintime=wintime, steptime=steptime, dither=dither)
    aspec = audspec(pspec, sr; nfilts=nbands, fbtype=fbtype, minfreq=minfreq,
                    maxfreq=maxfreq, sumpower=sumpower, bwidth=bwidth)
    if usecmp
        #  PLP-like weighting/compression
        aspec = postaud(aspec, maxfreq, fbtype)
    end
    if modelorder > 0
        if dcttype != 1
            ArgumentError("Sorry, modelorder>0 and dcttype ≠ 1 is not implemented")
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

mfcc(x::AbstractVector{<:AbstractFloat}, sr::Real=16000.0; args...) = mfcc(Vector(x), sr; args...)
mfcc(x::AbstractMatrix{<:AbstractFloat}, sr::Real=16000.0; args...) = @distributed (tuple) for i in axes(x, 2) mfcc(x[:, i], sr; args...) end


## default feature configurations, :rasta, :htk, :spkid_toolkit, :wbspeaker
## With optional extra agrs... you can specify more options
function mfcc(x::AbstractVector{<:AbstractFloat}, sr::AbstractFloat, defaults::Symbol; args...)
    if defaults == :rasta
        mfcc(x, sr; lifterexp=0.6, sumpower=true, nbands=40, dcttype=2, fbtype=:mel, args...)
    elseif defaults ∈ (:spkid_toolkit, :nbspeaker)
        mfcc(x, sr; lifterexp=0.6, sumpower=true, nbands=30, dcttype=2, fbtype=:mel, minfreq=130., maxfreq=3900., numcep=20, args...)
    elseif defaults == :wbspeaker
        mfcc(x, sr; lifterexp=0.6, sumpower=true, nbands=63, dcttype=2, fbtype=:mel, minfreq=62.5, maxfreq=7937.5, numcep=20, args...)
    elseif defaults == :htk
        mfcc(x, sr; args...)
    else
        ArgumentError(string("Unknown set of defaults: ", defaults))
    end
end

## our features run down with time, this is essential for the use of DSP.filt()
function deltas(x::AbstractMatrix{T}, w::Int=9) where {T<:AbstractFloat}
    nr, nc = size(x)
    if iszero(nr) || w <= 1
        return x
    end
    hlen = w ÷ 2
    w = 2hlen + 1                 # make w odd
    win = collect(convert(T, hlen):-1:-hlen)
    x1 = x[1:1, :]
    xend = x[end:end,:]
    xx = vcat(repeat(x1, hlen), x, repeat(xend, hlen)) ## take care of boundaries
    norm = 3 / (hlen * w * (hlen + 1))
    delta_v = filt(PolynomialRatio(win, [1.]), xx)[2hlen .+ (1:nr), :]
    @. delta_v *= norm
    return delta_v
end

sortperm_along(a::AbstractArray, dim::Int) =
(v=similar(a, Int, size(a, dim)); mapslices(x->sortperm!(v, x), a; dims=dim))

erfinvtabs = Dict{Int, Vector{Float64}}()
function erfinvtab(wl::Int)
    global erfinvtabs
    if !haskey(erfinvtabs, wl)
        erfinvtabs[wl] = @. √2 * erfinv(2(1:wl) / (wl + 1) - 1)
    end
    return erfinvtabs[wl]
end

function warpstats(x::AbstractMatrix{<:Real}, w::Int=399)
    nx = size(x, 1)
    wl = min(w, nx)
    if nx < w
        rank = sortperm_along(x, 1)
    else
        rank = similar(x, Int)
        hw = round(Int, (wl+1) / 2)
        for j in axes(x, 2), i in 1:nx  # operations in outer loop over columns, better for memory cache
            s = max(1, i - hw + 1)
            e = s + w - 1
            if e > nx
                d = e - nx
                e -= d
                s -= d
            end
            r = 1
            for k in s:e
                if x[i, j] > x[k, j]
                    r += 1
                end
            end
            rank[i, j] = r
        end
    end
    return rank, erfinvtab(wl)
end

function warp(x::AbstractMatrix{<:Real}, w::Int=399)
    rank, erfinvtab = warpstats(x, w)
    return erfinvtab[rank]
end
warp(x::AbstractVector{<:Real}, w::Int=399) = warp(reshape(x, :, 1), w)

function WarpedArray(x::AbstractMatrix{<:Real}, w::Int)
    rank, erfinvtab = warpstats(x, w)
    WarpedArray(rank, erfinvtab)
end

## mean and variance normalization
znorm(x::AbstractArray, dim::Int=1) = znorm!(copy(x), dim)
function znorm!(x::AbstractArray, dim::Int=1)
    mean_x = mean(x; dims=dim)
    std_x = std(x; dims=dim)
    @. x = (x - mean_x) / std_x
    x
end

## short-term mean and variance normalization
function stmvn(x::AbstractVector{T}, w::Int=399) where T
    y = similar(x, promote_type(Float64, T))
    nx = length(x)
    nx ≤ 1 || w <= 1 && return copy!(y, x)
    hw = w ÷ 2 ## effectively makes `w` odd...
    len_w = min(nx, hw + 1)
    v = Iterators.take(x, len_w)
    ## initialize sum x and sum x^2
    sx = sum(v) + hw * first(x)
    sxx = sum(x -> x^2, v) + hw * first(x)^2
    if hw + 1 > nx
        sx += (hw + 1 - nx) * last(x)
        sxx += (hw + 1 - nx) * last(x)^2
    end
    for i in eachindex(x, y)
        μ = sx / w
        σ = sqrt((sxx - μ * sx) / (w - 1))
        y[i] = (x[i] - μ) / σ
        mi = max(i - hw, firstindex(x))
        ma = min(i + hw + 1, lastindex(x))
        sx += x[ma] - x[mi]
        sxx += x[ma]^2 - x[mi]^2
    end
    return y
end

stmvn(m::AbstractMatrix, w::Int=399, dim::Int=1) = mapslices(x->stmvn(x, w), m; dims=dim)

## Shifted Delta Coefficients by copying
function sdc(x::AbstractMatrix{T}, n::Int=7, d::Int=1, p::Int=3, k::Int=7) where {T<:AbstractFloat}
    nx, nfea = size(x)
    n ≤ nfea || DomainError(n, "Not enough features for n")
    hnfill = (k-1) * p / 2
    nfill1, nfill2 = floor(Int, hnfill), ceil(Int, hnfill)
    xx = vcat(zeros(T, nfill1, n), deltas(x[:,1:n], 2d+1), zeros(T, nfill2, n))
    y = zeros(T, nx, n*k)
    for (dt, offset) in zip(0:p:p*k-1, 0:n:n*k-1)
        y[:, offset.+(1:n)] = xx[dt.+(1:nx), :]
    end
    return y
end
