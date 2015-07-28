## Feacalc.jl Full implementation of feature calculation for speech files. 
## (c) 2013 David A. van Leeuwen

## This program is free software: you can redistribute it and/or modify
##     it under the terms of the GNU General Public License as published by
##     the Free Software Foundation, version 3 of the License.

##     This program is distributed in the hope that it will be useful,
##     but WITHOUT ANY WARRANTY; without even the implied warranty of
##     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##     GNU General Public License for more details.

##     You should have received a copy of the GNU General Public License
##     along with this program.  If not, see <http://www.gnu.org/licenses/>. 

## Feacalc.  Feature calculation as used for speaker and language recognition. 

nrow(x) = size(x,1)
ncol(x) = size(x,2)

## compute features according to standard settingsm directly from a wav file. 
## this does channel at the time. 
function feacalc(wavfile::String; method=:sox, augtype=:ddelta, normtype=:warp, sadtype=:energy, defaults=:spkid_toolkit, dynrange::Real=30., nwarp::Int=399, chan=:mono)
    if method == :wav
        (x, sr) = wavread(wavfile)
    elseif method == :sox
        (x, sr) = soxread(wavfile)
    elseif method == :sphere
        (x, sr) = sphread(wavfile)
    end
    sr = float64(sr)       # more reasonable sr
    feacalc(x; augtype=augtype, normtype=normtype, sadtype=sadtype, defaults=defaults, dynrange=dynrange, nwarp=nwarp, chan=chan, sr=sr, source=wavfile)
end

## assume we have an array already
function feacalc(x::Array; augtype=:ddelta, normtype=:warp, sadtype=:energy, defaults=:spkid_toolkit, dynrange::Real=30., nwarp::Int=399, chan=:mono, sr::FloatingPoint=8000.0, source=":array")
    if ndims(x)>1
        nsamples, nchan = size(x)
        if chan == :mono
            x = vec(mean(x, 2))            # averave multiple channels for now
        elseif in(chan, [:a, :b])
            channum = findin([:a, :b], [chan])
            x = vec(x[:,channum])
        elseif isa(chan, Integer) 
            if !(chan in 1:nchan)
                error("Bad channel specification: ", chan)
            end
            x = vec(x[:,chan])
            chan=[:a, :b][chan]
        else
            error("Unknown channel specification: ", chan)
        end
    else
        nsamples, nchan = length(x), 1
    end
    ## save some metadata
    meta = ["nsamples" => nsamples, "sr" => sr, "source" => source, "nchan" => nchan,
            "chan" => chan]
    preemp = 0.97
    preemp ^= 16000. / sr

    ## basic features
    (m, pspec, params) = mfcc(x, sr, defaults)
    meta["totnframes"] = nrow(m)
    
    ## augment features
    if augtype==:delta || augtype==:ddelta
        d = deltas(m)
        if augtype==:ddelta
            dd = deltas(d)
            m = hcat(m, d, dd)
        else 
            m = hcat(m, d)
        end
    elseif augtype==:sdc
        m = sdc(m)
    end
    meta["augtype"] = augtype

    if nrow(m)>0
        if sadtype==:energy
            ## integrate power
            deltaf = size(pspec,2) / (sr/2)
            minfreqi = iround(300deltaf)
            maxfreqi = iround(4000deltaf)
            power = 10log10(sum(pspec[:,minfreqi:maxfreqi], 2))
            
            maxpow = maximum(power)
            speech = find(power .> maxpow - dynrange)
            params["dynrange"] = dynrange
        elseif sadtype==:none
            speech = [1:nrow(m)]
        end
    else
        speech=Int[]
    end
    meta["sadtype"] = sadtype
    ## perform SAD
    m = m[speech,:]
    meta["speech"] = uint32(speech)
    meta["nframes"] , meta["nfea"] = size(m)
    
    ## normalization
    if nrow(m)>0
        if normtype==:warp
            m = warp(m, nwarp)
            params["warp"] = nwarp          # the default
        elseif normtype==:mvn
            if nrow(m)>1
                znorm!(m,1)
            else
                fill!(m, 0)
            end
        end
        meta["normtype"] = normtype
    end
    return(float32(m), meta, params)
end

## When called with a specific application in mind, call with two arguments
function feacalc(wavfile::String, application::Symbol; chan=:mono, method=:sox)
    if (application==:speaker)
        feacalc(wavfile; defaults=:spkid_toolkit, chan=chan, method=method)
    elseif application==:wbspeaker
        feacalc(wavfile; defaults=:wbspeaker, chan=chan, method=method)
    elseif (application==:language)
        feacalc(wavfile; defaults=:rasta, nwarp=299, augtype=:sdc, chan=chan, 
                method=method)
    elseif (application==:diarization)
        feacalc(wavfile; defaults=:rasta, sadtype=:none, normtype=:mvn, 
                augtype=:none, chan=chan, method=method)
    else
        error("Unknown application ", application)
    end
end

function sad(pspec::Matrix{Float64}, sr::Float64, method=:energy; dynrange::Float64=30.)
    deltaf = size(pspec,2) / (sr/2)
    minfreqi = iround(300deltaf)
    maxfreqi = iround(4000deltaf)
    power = 10log10(sum(pspec[:,minfreqi:maxfreqi], 2))
    maxpow = maximum(power)
    speech = find(power .> maxpow - dynrange)
end

## listen to SAD
function sad(wavfile::String, speechout::String, silout::String; dynrange::Float64=30.)
    (x, sr, nbits) = wavread(wavfile)
    sr = float64(sr)               # more reasonable sr
    x = vec(mean(x, 2))            # averave multiple channels for now
    (m, pspec, meta) = mfcc(x, sr; preemph=0)
    sp = sad(pspec, sr, dynrange=dynrange)
    sl = iround(meta["steptime"] * sr)
    xi = falses(size(x))
    for (i in sp)
        xi[(i-1)*sl+(1:sl)] = true
    end
    y = x[find(xi)]
    wavwrite(y, speechout, Fs=sr, nbits=nbits, compression=WAVE_FORMAT_PCM)
    y = x[find(!xi)]
    wavwrite(y, silout, Fs=sr, nbits=nbits, compression=WAVE_FORMAT_PCM)
end

## this should probably be called soxread...
function soxread(file)
    nch = int(readall(`soxi -c $file`))
    sr = int(readall(`soxi -r $file`))
    sox = `sox $file -t raw -e signed -r $sr -b 16 -`
    fd, proc = open(sox, "r")
    x = Int16[]
    while !eof(fd)
        push!(x, read(fd, Int16))
    end
    ns = div(length(x), nch)
    reshape(x, nch, ns)' / (1<<15), sr
end

## similarly using sphere tools
function sphread(file)
    nch = int(readall(`h_read -n -F channel_count $file`))
    sr = int(readall(`h_read -n -F sample_rate $file`))
    sphere = `w_decode -o pcm $file -` |> `h_strip - - `
    fd, proc = readsfrom(sphere)
    x = Int16[]
    while !eof(fd)
        push!(x, read(fd, Int16))
    end
    ns = div(length(x), nch)
    reshape(x, nch, ns)' / (1<<15), sr
end
    
