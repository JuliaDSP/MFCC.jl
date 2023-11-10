## io.jl
## (c) 2014 David A. van Leeuwen
##
## i/o functions for speech feature data.
## For compatibility reasons, we do not use JLD but encode everything in HDF5 types
## Is that a valid argument?  I don't know

import HDF5

## encode non-HDF5 types in the key by adding type indicator---a poorman's solution
HDF5.write(fd::HDF5.File, s::AbstractString, sym::Symbol) = write(fd, string(s,":Symbol"), string(sym))

## always save data in Float32
## the function arguments are the same as the output of feacalc
## x: the MFCC data
## meta: a dict with parameters about the data, nsamples, nframes, totnframes (before sad), ...
## params: a dict with parameters about the feature extraction itself.
function feasave(file::AbstractString, x::AbstractMatrix{<:AbstractFloat}; meta::Dict=Dict(), params::Dict=Dict())
    dir = dirname(file)
    if !(isempty(dir) || isdir(dir))
        mkpath(dir)
    end
    HDF5.h5open(file, "w") do fd
        fd["features/data"] = map(Float32, x)
        for (k, v) in meta
            fd[string("features/meta/", k)] = v
        end
        for (k, v) in params
            fd[string("features/params/", k)] = v
        end
    end
end

## JLD version of the same
## FileIO.save(file::AbstractString, x::Matrix)

## the reverse encoding of Julia types.
function retype(d::Dict{<:AbstractString, T} where T)
    r = Dict{Symbol, Any}()
    for (k, v) in d
        if (m=match(r"(.*):Symbol", k)) !== nothing
            r[Symbol(m.captures[1])] = Symbol(v)
        else
            r[Symbol(k)] = v
        end
    end
    r
end

## Try to handle missing elements in the hdf5 file more gracefully
function h5check(obj, name, content)
    HDF5.haskey(obj, content) || error('"', name, '"', " does not contain ", '"', content, '"')
    obj[content]
end

## Currently we always read the data in Float64
function feaload(file::AbstractString; meta::Bool=false, params::Bool=false)
    HDF5.h5open(file, "r") do fd
        f = h5check(fd, file, "features")
        fea = map(Float64, read(h5check(f, "features", "data")))
        if isempty(fea)             # "no data"
            m = read(h5check(f, "features", "meta"))
            fea = zeros(0, m["nfea"])
        end
        if !(meta || params)
            return fea
        end
        res = Any[fea]
        if meta
            push!(res, retype(read(h5check(f, "features", "meta"))))
        end
        if params
            push!(res, retype(read(h5check(f, "features", "params"))))
        end
        Tuple(res)
    end
end

## helper function to quickly determine the size of a feature file
function feasize(file::AbstractString)
    HDF5.h5open(file, "r") do fd
        f = h5check(fd, file, "features")
        size(h5check(f, "features", "data"))
    end
end
