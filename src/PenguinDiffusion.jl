module PenguinDiffusion

using LinearAlgebra
using LinearSolve
using SparseArrays

using CartesianGeometry
using CartesianOperators
using PenguinSolverCore

include("types.jl")
include("types_2phase.jl")
include("build.jl")
include("build_2phase.jl")
include("reduction.jl")
include("reduction_2phase.jl")
include("unsteady_matrixfree.jl")
include("unsteady_block.jl")
include("rhs.jl")
include("rhs_2phase.jl")
include("rebuild.jl")
include("rebuild_2phase.jl")
include("io.jl")
include("io_2phase.jl")
include("updaters.jl")
include("updaters_2phase.jl")
include("steady.jl")

export DiffusionProblem
export DiffusionSystem
export TwoPhaseDiffusionProblem
export TwoPhaseDiffusionSystem
export build_system
export build_matrixfree_system
export enable_matrixfree_unsteady!
export full_state
export RobinGUpdater, RobinABUpdater, BoxDirichletUpdater, KappaUpdater, SourceUpdater
export FluxJumpGUpdater, FluxJumpBUpdater, ScalarJumpGUpdater, ScalarJumpAlphaUpdater
export Kappa1Updater, Kappa2Updater
export BoxDirichletUpdater1, BoxDirichletUpdater2
export Source1Updater, Source2Updater
export steady_linear_problem, steady_solve
export unsteady_block_matrix, unsteady_block_solve
export diphasic_unsteady_block_matrix, diphasic_unsteady_block_solve

end
