## MFCCs.jl
## (c) 2013-2014 David A. van Leeuwen, (c) 2005--2012 Dan Ellis

## Recoded from / inspired by melfcc from Dan Ellis's rastamat package. 

module MFCCs

export powspec, audspec, postaud, lifter
export mfcc, deltas, warp, sdc, znorm, znorm!

using DSP

include("rasta.jl")
include("mfccs.jl")

end
