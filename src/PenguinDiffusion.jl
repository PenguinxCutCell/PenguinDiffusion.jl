module PenguinDiffusion

using LinearAlgebra
using SparseArrays

using CartesianGeometry
using CartesianOperators
using PenguinSolverCore

include("types.jl")
include("build.jl")
include("reduction.jl")
include("rhs.jl")
include("rebuild.jl")
include("io.jl")
include("updaters.jl")

export DiffusionProblem
export DiffusionSystem
export build_system
export full_state
export RobinGUpdater, KappaUpdater, SourceUpdater

end
