## (c) 2013--2014 David A. van Leeuwen, (c) 2005--2012, Dan Ellis

## Freely adapted from Dan Ellis's rastamat package.
## We've kept routine names the same, but the interface has changed a bit.

## we haven't implemented rasta filtering, yet, in fact.  These routines are a minimum for
## encoding HTK-style mfccs

# powspec tested against octave with simple vectors

using DSP
using FFTW
using LinearAlgebra

function powspec(x::AbstractVector{<:AbstractFloat}, sr::Real=8000.0; wintime=0.025, steptime=0.01, dither=true)
    nwin = round(Integer, wintime * sr)
    nstep = round(Integer, steptime * sr)

    nfft = nextpow(2, nwin)
    window = hamming(nwin)      # overrule default in specgram which is hanning in Octave
    noverlap = nwin - nstep

    y = spectrogram(x .* (1<<15), nwin, noverlap, nfft=nfft, fs=sr, window=window, onesided=true).power
    ## for compability with previous specgram method, remove the last frequency and scale
    y = y[1:end-1, :] ##  * sumabs2(window) * sr / 2
    y .+= dither * nwin / (sum(abs2, window) * sr / 2) ## OK with julia 0.5, 0.6 interpretation as broadcast!

    return y
end

# audspec tested against octave with simple vectors for all fbtypes
function audspec(x::AbstractMatrix{<:AbstractFloat}, sr::Real=16000.0;
                 nfilts=ceil(Int, hz2bark(sr / 2)), fbtype=:bark,
                 minfreq=0., maxfreq=sr/2, sumpower::Bool=true, bwidth=1.0)
    nfreqs, nframes = size(x)
    nfft = 2(nfreqs-1)
    if fbtype == :bark
        wts = fft2barkmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq)
    elseif fbtype == :mel
        wts = fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq)
    elseif fbtype == :htkmel
        wts = fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq,
                        htkmel=true, constamp=true)
    elseif fbtype == :fcmel
        wts = fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq,
                        htkmel=true, constamp=false)
    else
        error("Unknown filterbank type ", fbtype)
    end
    wts = wts[:, 1:nfreqs]
    if sumpower
        return wts * x
    else
        return (wts * sqrt.(x)).^2
    end
end

function fft2barkmx(nfft::Int, nfilts::Int; sr=8000.0, width=1.0, minfreq=0., maxfreq=sr/2)
    hnfft = nfft >> 1
    minbark = hz2bark(minfreq)
    nyqbark = hz2bark(maxfreq) - minbark
    wts = zeros(nfilts, nfft)
    stepbark = nyqbark / (nfilts-1)
    binbarks = hz2bark.((0:hnfft) * sr / nfft)
    for i in 1:nfilts
        midbark = minbark + (i-1) * stepbark
        lof = (binbarks .- midbark) / width .- 0.5
        hif = (binbarks .- midbark) / width .+ 0.5
        logwts = min.(0, hif, -2.5lof)
        wts[i, 1:1+hnfft] = 10.0.^logwts
    end
    return wts
end

## Hynek's formula
hz2bark(f) = 6 * asinh(f / 600)
bark2hz(bark) = 600 * sinh(bark / 6)

function slope_gen(fs, fftfreq)
    f1, f2, f3 = fs
    # lower and upper slopes for all bins
    loslope = (fftfreq - f1) / (f2 - f1)
    hislope = (f3 - fftfreq) / (f3 - f2)
    # then intersect them with each other and zero
    max(0, min(loslope, hislope))
end

function fft2melmx(nfft::Int, nfilts::Int; sr=8000.0, width=1.0, minfreq=0.0,
                   maxfreq=sr/2, htkmel::Bool=false, constamp::Bool=false)
    wts = zeros(nfilts, nfft)
    lastind = (nfft>>1)
    # Center freqs of each DFT bin
    fftfreqs = collect(0:lastind-1) / nfft * sr;
    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfreq, htkmel);
    maxmel = hz2mel(maxfreq, htkmel);
    binfreqs = @. mel2hz(minmel + (0:(nfilts+1)) / (nfilts + 1) * (maxmel - minmel), htkmel);
##    binbin = iround(binfrqs/sr*(nfft-1));

    for i in 1:nfilts
        fs = binfreqs[i], binfreqs[i+1], binfreqs[i+2]
        # scale by width
        fs = @. fs[2] + (fs - fs[2])width
        for j in eachindex(fftfreqs)
            wts[i, j] = slope_gen(fs, fftfreqs[j])
        end
    end

    if !constamp
        ## unclear what this does...
        ## Slaney-style mel is scaled to be approx constant E per channel
        @. wts = 2 / ((binfreqs[3:nfilts+2]) - binfreqs[1:nfilts]) * wts
        # Make sure 2nd half of DFT is zero
        wts[:, lastind+1:nfft] .= 0.
    end
    return wts
end

function hz2mel(f::AbstractVector{<:AbstractFloat}, htk::Bool=false)
    if htk
        return @. 2595 * log10(1 + f / 700)
    else
        f0 = 0.0
        fsp = 200 / 3
        brkfrq = 1000.0
        brkpt = (brkfrq - f0) / fsp
        logstep = exp(log(6.4) / 27)
        linpts = f .< brkfrq
        z = zeros(axes(f))      # prevent InexactError() by making these Float64
        @. z[linpts] = f[linpts] / brkfrq / log(logstep)
        @. z[!linpts] = brkpt + log(f[!linpts] / brkfrq) / log(logstep)
    end
    return z
end
hz2mel(f::AbstractFloat, htk::Bool=false)  = hz2mel([f], htk)[1]

function mel2hz(z::AbstractFloat, htk::Bool=false)
    if htk
        f = @. 700 * (10 ^ (z / 2595) - 1)
    else
        f0 = 0.0
        fsp = 200 / 3
        brkfrq = 1000.0
        brkpt = (brkfrq - f0) / fsp
        logstep = exp(log(6.4) / 27)
        linpt = z < brkpt
        f = linpt ? f0 + fsp * z : brkfrq * exp(log(logstep) * (z - brkpt))
    end
    return f
end

function postaud(x::Matrix{T}, fmax::Real, fbtype=:bark, broaden=false) where {T<:AbstractFloat}
    nbands, nframes = size(x)
    nfpts = nbands + 2broaden
    if fbtype == :bark
        bandcfhz = bark2hz(range(0, stop=hz2bark(fmax), length=nfpts))
    elseif fbtype == :mel
        bandcfhz = mel2hz.(range(0, stop=hz2mel(fmax), length=nfpts))
    elseif fbtype == :htkmel || fbtype == :fcmel
        bandcfhz = mel2hz.(range(0, stop=hz2mel(fmax, true), length=nfpts),1);
    else
        error("Unknown filterbank type")
    end
    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[1+broaden:nfpts-broaden];
    # Hynek's magic equal-loudness-curve formula
    fsq = bandcfhz.^2
    ftmp = fsq + 1.6e5
    eql = @. ((fsq / ftmp)^2) * ((fsq + 1.44e6) / (fsq + 9.61e6))
    # weight the critical bands
    z = eql .* x
    # cube root compress
    @. z = cbrt(z)
    # replicate first and last band (because they are unreliable as calculated)
    if broaden
        z = z[[1, 1:nbands, nbands], :];
    else
        z = z[[2, 2:(nbands-1), nbands-1], :]
    end
    return z
end

function dolpc(x::Array{T}, modelorder::Int=8) where {T<:AbstractFloat}
    nbands, nframes = size(x)
    r = real(ifft(vcat(x, x[collect(nbands-1:-1:2), :]), 1)[1:nbands, :])
    # Find LPC coeffs by durbin
    y, e = levinson(r, modelorder)
    # Normalize each poly by gain
    y ./= e
end

function lpc2cep(a::Array{T}, ncep::Int=0) where {T<:AbstractFloat}
    nlpc, nc = size(a)
    order = nlpc - 1
    if ncep == 0
        ncep = nlpc
    end
    c = zeros(ncep, nc)
    # Code copied from HSigP.c: LPC2Cepstrum
    # First cep is log(Error) from Durbin
    c[1,:] = -log.(a[1, :])
    # Renormalize lpc A coeffs
    a ./= a[1, :]
    for n in 2:ncep
        sum = zero(T)
        for m in 2:n
            sum += (n - m) * a[m, :] .* c[n - m + 1, :]
        end
        c[n, :] = -a[n, :] - sum / (n - 1)
    end
    return c
end

function spec2cep(spec::AbstractMatrix{<:AbstractFloat}, ncep::Int=13, dcttype::Int=2)
    # no discrete cosine transform option
    dcttype == -1 && return log.(spec)

    nr, nc = size(spec)
    dctm = similar(spec, ncep, nr)
    if 1 < dcttype < 4          # type 2,3
        for j in 1:nr, i in 1:ncep
            dctm[i, j] = cospi((i - 1) * (2j-1) / (2nr)) * √(2/nr)
        end
        if dcttype == 2
            @. dctm[1, :] /= √2
        end
    elseif dcttype == 4         # type 4
        for j in 1:nr, i in 1:ncep
            dctm[i, j] = 2cospi((i-1) * j/(nr+1))
        end
        for i in axes(dctm, 1)
            dctm[i, end] += (-1)^(i-1)
        end
        @. dctm[:, 1] += 1
        @. dctm /= 2(nr + 1)
    else                        # type 1
        for j in 1:nr, i in 1:ncep
            dctm[i, j] = cospi((i-1) * (j-1) / (nr - 1)) / (nr - 1)
        end
        @. dctm[:, [1, nr]] /= 2
    end
    # assume spec is not reused
    return dctm * map!(log, spec, spec)
end

function lifter(x::AbstractArray{<:AbstractFloat}, lift::Real=0.6, invs=false)
    ncep, nf = size(x)
    if iszero(lift)
        return x
    elseif lift > 0
        if lift > 10
            error("Lift number is too high (>10)")
        end
        liftw = pushfirst!((1:ncep-1).^lift, 1)
    else
        # Hack to support HTK liftering
        if !isa(lift, Integer)
            error("Negative lift must be interger")
        end
        lift = -lift            # strictly speaking unnecessary...
        liftw = @. 1 + lift / 2 * sinpi((0:ncep-1) / lift)
    end
    if invs
        @. liftw = inv(liftw)
    end
    y = broadcast(*, x, liftw)
    return y
end

## Freely after octave's implementation, by Paul Kienzle <pkienzle@users.sf.net>
## Permission granted to usee this in a MIT license on 20 dec 2013 by the author Paul Kienzle:
## "You are welcome to move my octave code from GPL to MIT like core Julia."
## untested
## only returns a, v
function levinson(acf::Vector{T}, p::Int) where {T<:Real}
    if length(acf) < 1
        error("empty autocorrelation function")
    end
    if p < 0
        error("negative model order")
    end
    if p < 100
        ## direct solution [O(p^3), but no loops so slightly faster for small p]
        ## Kay & Marple Eqn (2.39)
        R = toeplitz(acf[1:p], conj(acf[1:p]))
        a = R \ -acf[2:p+1]
        pushfirst!(a, 1)
        v = real(a.*conj(acf[1:p+1]))
    else
        ## durbin-levinson [O(p^2), so significantly faster for large p]
        ## Kay & Marple Eqns (2.42-2.46)
        ref = zeros(p)
        g = -acf[2] / acf[1]
        a = [g]
        v = real((1 - abs2(g)) * acf[1])
        ref[1] = g
        for t=2:p
            g = -(acf[t+1] + a ⋅ acf[t:-1:2]) / v
            a = [a + g * conj(a[t-1:-1:1]), g]
            v *= 1 - abs2(g)
            ref[t] = g
        end
        pushfirst!(a, 1)
    end
    return a, v
end

function levinson(acf::Array{T}, p::Int) where {T<:Real}
    nr, nc = size(acf)
    a = zeros(p + 1, nc)
    v = zeros(p + 1, nc)
    for i in 1:nc
        a[:,i], v[:,i] = levinson(acf[:,i], p)
    end
    return a, v
end

## Freely after octave's implementation, ver 3.2.4, by jwe && jh
## skipped sparse implementation
function toeplitz(c::Vector{T}, r::Vector{T}=c) where {T<:Real}
    nc = length(r)
    nr = length(c)
    res = zeros(T, nr, nc)
    if iszero(nc) || iszero(nr)
        return res
    end
    if r[1] != c[1]
        ## warn
    end
    if typeof(c) <: Complex
        conj!(c)
    end
    ## if issparse(c) && ispsparse(r)
    data = [r[end:-1:2], c]
    for (i, start) in zip(1:nc, nc:-1:1)
        res[:,i] = data[start:start+nr-1]
    end
    return res
end
