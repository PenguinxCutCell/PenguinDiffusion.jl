using Documenter
using PenguinDiffusion

makedocs(
    modules = [PenguinDiffusion],
    authors = "PenguinxCutCell contributors",
    sitename = "PenguinDiffusion.jl",
    format = Documenter.HTML(
        canonical = "https://PenguinxCutCell.github.io/PenguinDiffusion.jl",
        repolink = "https://github.com/PenguinxCutCell/PenguinDiffusion.jl",
        collapselevel = 2,
    ),
    pages = [
        "Home" => "index.md",
        "Equations" => "equations.md",
        "Boundary Conditions" => "boundary-conditions.md",
        "Numerics" => "numerics.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
        "Validation" => "validation.md",
        "Design Notes" => "design.md",
    ],
    pagesonly = true,
    warnonly = false,
    remotes = nothing,
)

if get(ENV, "CI", "") == "true"
    deploydocs(
        repo = "github.com/PenguinxCutCell/PenguinDiffusion.jl",
        push_preview = true,
    )
end
