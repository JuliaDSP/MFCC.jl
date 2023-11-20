## Feacalc.jl Full implementation of feature calculation for speech files.
## (c) 2013-2016 David A. van Leeuwen

## Feacalc.  Feature calculation as used for speaker and language recognition.

nrow(x) = size(x, 1)
ncol(x) = size(x, 2)

"""
compute features according to standard settings directly from a wav file.
this does channel at the time.
"""
function feacalc(wavfile; method=:wav, kwargs...)
    x, sr = if method == :wav
        wavread(wavfile)
    elseif method == :sox
        soxread(wavfile)
    elseif method == :sphere
        sphread(wavfile)
    else
        throw(ArgumentError(string("Method not supported: ", method)))
    end
    sr = Float64(sr)       # more reasonable sr
    feacalc(x; sr=sr, source=wavfile, kwargs...)
end

## assume we have an array already
function feacalc(x::AbstractVecOrMat; augtype=:ddelta, normtype=:warp, sadtype=:energy,
                 dynrange::Real=30., nwarp::Int=399, chan=:mono, sr::AbstractFloat=8000.0,
                 source=":array", defaults=:nbspeaker, mfccargs...)
    if x isa AbstractMatrix
        nsamples, nchan = size(x)
        if chan == :mono
            x = vec(mean(x; dims=2))            # average multiple channels for now
        elseif chan in (:a, :b)
            channum = chan == :b
            x = x[:, begin+channum]
        elseif chan isa Integer
            if !(chan in 1:nchan)
                throw(DomainError(chan, "Bad channel specification"))
            end
            x = x[:, begin+chan-1]
            chan = chan in 1:2 ? (:a, :b)[chan] : chan
        else
            throw(ArgumentError(string("Unknown channel specification: ", chan)))
        end
    else
        nsamples, nchan = length(x), 1
    end
    ## save some metadata
    meta = Dict{Symbol, Any}(
        :nsamples => nsamples, :sr => sr, :source => source,
        :nchan => nchan, :chan => chan
        )
    preemph = 0.97
    preemph ^= 16000. / sr

    ## basic features
    m, pspec, params = mfcc(x, sr, defaults; preemph=preemph, mfccargs...)
    meta[:totnframes] = nrow(m)

    ## augment features
    if augtype in (:delta, :ddelta)
        d = deltas(m)
        if augtype == :ddelta
            dd = deltas(d)
            m = hcat(m, d, dd)
        else
            m = hcat(m, d)
        end
    elseif augtype == :sdc
        m = sdc(m)
    end
    meta[:augtype] = augtype

    if !isempty(m)
        if sadtype == :energy
            speech = sad(pspec, sr; dynrange=dynrange)
            params[:dynrange] = dynrange
            ## perform SAD
            m = m[speech,:]
            meta[:speech] = map(UInt32, speech)
        elseif sadtype == :none
            nothing
        end
    end
    meta[:sadtype] = sadtype
    meta[:nframes], meta[:nfea] = size(m)

    ## normalization
    if !isempty(m)
        if normtype == :warp
            m = warp(m, nwarp)
            params[:warp] = nwarp          # the default
        elseif normtype == :mvn
            if nrow(m) > 1
                znorm!(m, 1)
            else
                fill!(m, 0)
            end
        end
        meta[:normtype] = normtype
    end
    return (map(Float32, m), meta, params)
end

## When called with a specific application in mind, call with two arguments
function feacalc(wavfile, application::Symbol; kwargs...)
    if application in (:speaker, :nbspeaker)
        feacalc(wavfile; defaults=:nbspeaker, kwargs...)
    elseif application == :wbspeaker
        feacalc(wavfile; defaults=:wbspeaker, kwargs...)
    elseif application == :language
        feacalc(wavfile; defaults=:rasta, nwarp=299, augtype=:sdc, kwargs...)
    elseif application == :diarization
        feacalc(wavfile; defaults=:rasta, sadtype=:none, normtype=:mvn, augtype=:none, kwargs...)
    else
        throw(ArgumentError(string("Unknown application: ", application)))
    end
end

function sad(pspec::AbstractMatrix, sr::T, method=:energy; dynrange::T=30.) where T<:Float64
    ## integrate power
    deltaf = size(pspec, 2) / (sr/2)
    minfreqi = round(Int, 300deltaf)
    maxfreqi = round(Int, 4000deltaf)
    summed_pspec = vec(sum(view(pspec, :, minfreqi:maxfreqi); dims=2))
    power = @. 10log10(summed_pspec)
    maxpow = maximum(power)
    speech = findall(>(maxpow - dynrange), power)
    return speech
end

## listen to SAD
function sad(wavfile, speechout, silout; dynrange::Float64=30.)
    x, sr, nbits = wavread(wavfile)
    sr = Float64(sr)                            # more reasonable sr
    mx::Vector{Float64} = vec(mean(x; dims=2))  # average multiple channels for now
    m, pspec, meta = mfcc(mx, sr; preemph=0)
    sp = sad(pspec, sr; dynrange=dynrange)
    sl = round(Int, meta[:steptime] * sr)
    xi = falses(axes(mx))
    for i in @view sp[begin:end-1]
        z = (i - 1) * sl
        @. xi[z+1:z+sl] = true
    end
    z = (sp[end] - 1) * sl
    xi[z+1:min(z+sl, lastindex(mx))] .= true
    wavwrite(mx[xi], speechout, Fs=sr, nbits=nbits, compression=WAVE_FORMAT_PCM)
    wavwrite(mx[.!xi], silout, Fs=sr, nbits=nbits, compression=WAVE_FORMAT_PCM)
end

## this should probably be called soxread...
function soxread(file)
    nch = parse(Int, read(`soxi -c $file`, String))
    sr = parse(Int, read(`soxi -r $file`, String))
    sox = `sox $file -t raw -e signed -r $sr -b 16 -`
    x = Int16[]
    open(sox, "r") do fd, proc
        while !eof(fd)
            push!(x, read(fd, Int16))
        end
    end
    ns = length(x) ÷ nch
    reshape(x, nch, ns)' / (1<<15), sr
end

## similarly using sphere tools
function sphread(file)
    nch = parse(Int, read(`h_read -n -F channel_count $file`, String))
    sr = parse(Int, read(`h_read -n -F sample_rate $file`, String))
    open(pipeline(`w_decode -o pcm $file -`, `h_strip - - `), "r") do fd
        x = Int16[]
        while !eof(fd)
            push!(x, read(fd, Int16))
        end
        ns = length(x) ÷ nch
        return reshape(x, nch, ns)' / (1<<15), sr
    end
end
