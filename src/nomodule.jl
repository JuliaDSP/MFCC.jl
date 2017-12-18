## for development

using DSP
using HDF5

if !isdefined(:WarpedArray)
    include("types.jl")
end

include("rasta.jl")
include("mfccs.jl")
include("warpedarray.jl")
include("feacalc.jl")
include("io.jl")
