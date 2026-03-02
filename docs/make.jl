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
        "API" => "api.md",
        "Examples" => "examples.md",
        "Algorithms" => "algorithms.md",
        "Diffusion Models" => "diffusion.md",
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
