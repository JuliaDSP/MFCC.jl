## MFCC.jl
## (c) 2013-2015 David A. van Leeuwen, (c) 2005--2012 Dan Ellis

## Recoded from / inspired by melfcc from Dan Ellis's rastamat package. 

module MFCC

export powspec, audspec, postaud, lifter
export mfcc, deltas, warp, sdc, znorm, znorm!
export save, load
export feacalc

using DSP
using HDF5

include("rasta.jl")
include("mfccs.jl")
include("feacalc.jl")
include("io.jl")

end
