## io.jl
## (c) 2014 David A. van Leeuwen
##
## i/o functions for speech feature data.
## For compatibility reasons, we do not use JLD but encode everything in HDF5 types
## Is that a valid argument?  I don't know

import FileIO
#import JLD

## encode non-HDF5 types in the key by adding type indicator---a poorman's solution
HDF5.write(fd::HDF5File, s::AbstractString, b::Bool) = write(fd, string(s,":Bool"), Int8(b))
HDF5.write(fd::HDF5File, s::AbstractString, sym::Symbol) = write(fd, string(s,":Symbol"), string(sym))
HDF5.write(fd::HDF5File, s::AbstractString, ss::SubString) = write(fd, s, ascii(ss))

## always save data in Float32
## the functiona arguments are the same as the output of feacalc
## x: the MFCC data
## meta: a dict with parameters anout the data, nsamples, nframes, totnframes (before sad), ...
## params: a dict with parameters about the feature extraction itself.
function feasave(file::AbstractString, x::Matrix{T}; meta::Dict=Dict(), params::Dict=Dict()) where {T<:AbstractFloat}
    dir = dirname(file)
    if length(dir)>0 && !isdir(dir)
        mkpath(dir)
    end
    fd = h5open(file, "w")
    fd["features/data"] = map(Float32, x)
    for (k,v) in meta
        fd[string("features/meta/", k)] = v
    end
    for (k,v) in params
        fd[string("features/params/", k)] = v
    end
    close(fd)
end

## JLD version of the same
## FileIO.save(file::AbstractString, x::Matrix)

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

## Try to handle missing elements in the hdf5 file more gacefully
function h5check(obj, name, content)
    content ∈ names(obj) || error('"', name, '"', " does not contain ", '"', content, '"')
    obj[content]
end

## Currently we always read the data in float64
function feaload(file::AbstractString; meta=false, params=false)
    h5open(file, "r") do fd
        f = h5check(fd, file, "features")
        fea = map(Float64, read(h5check(f, "features", "data")))
        if length(fea)==0           # "no data"
            m = read(h5check(f, "features", "meta"))
            fea = zeros(0,m["nfea"])
        end
        if ! (meta || params)
            return fea
        end
        res = Any[fea]
        if meta
            push!(res, retype(read(h5check(f, "features", "meta"))))
        end
        if params
            push!(res, retype(read(h5check(f, "features", "params"))))
        end
        tuple(res...)
    end
end

## helper function to quickly determine the size of a feature file
function feasize(file::AbstractString)
    h5open(file, "r") do fd
        f = h5check(fd, file, "features")
        size(h5check(f, "features", "data"))
    end
end
