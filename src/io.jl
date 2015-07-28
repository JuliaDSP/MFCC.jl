## io.jl
## (c) 2014 David A. van Leeuwen
##
## i/o functions for speech feature data.  
## For compatibility reasons, we do not use JLD but encode everything in HDF5 types
## Is that a valid argument?  I don't know

## encode non-HDF5 types in the key by adding type indicator---a poorman's solution
HDF5.write(fd::HDF5File, s::ASCIIString, b::Bool) = write(fd, string(s,":Bool"), int8(b))
HDF5.write(fd::HDF5File, s::ASCIIString, sym::Symbol) = write(fd, string(s,":Symbol"), string(sym))
HDF5.write(fd::HDF5File, s::ASCIIString, ss::SubString) = write(fd, s, ascii(ss))

## always save data in Float32
## the functiona arguments are the same as the output of feacalc
## x: the MFCC data
## meta: a dict with parameters anout the data, nsamples, nframes, totnframes (before sad), ...
## params: a dict with parameters about the feature extraction itself. 
function save{T<:FloatingPoint}(file::String, x::Matrix{T}; meta::Dict=Dict(), params::Dict=Dict())
    dir = dirname(file)
    if length(dir)>0 && !isdir(dir)
        mkpath(dir)
    end
    fd = h5open(file, "w")
    fd["features/data"] = float32(x)
    for (k,v) in meta
        fd[string("features/meta/", k)] = v
    end
    for (k,v) in params
        fd[string("features/params/", k)] = v
    end
    close(fd)
end

## the reverse encoding of Julia types. 
function retype(d::Dict)
    r = Dict()
    for (k,v) in d
        if (m=match(r"^(.*):Bool$", k)) != nothing
            r[m.captures[1]] = v>0
        elseif (m=match(r"(.*):Symbol", k)) != nothing
            r[m.captures[1]] = symbol(v)
        else
            r[k] = v
        end
    end
    r
end

## but always read into float64 
function load(file::String; meta=false, params=false)
    fd = h5open(file, "r")
    fea = float64(read(fd["features/data"]))
    if length(fea)==0           # "no data"
        m = read(fd["features/meta"])
        fea = zeros(0,m["nfea"])
    end
    if ! (meta || params)
        return fea
    end
    res = Any[fea]
    if meta
        push!(res, retype(read(fd["features/meta"])))
    end
    if params
        push!(res, retype(read(fd["features/params"])))
    end
    tuple(res...)
end
