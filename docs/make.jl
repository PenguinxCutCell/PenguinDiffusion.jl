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
    # TODO: enable doctest once we curate fully deterministic snippets.
    doctest = false,
    pages = [
        "Home" => "index.md",
        "Theory" => "diffusion.md",
        "Steady" => "steady.md",
        "Unsteady (θ)" => "unsteady.md",
        "Moving Slabs" => "moving.md",
        "Interface Conditions" => "interface_conditions.md",
        "Algorithms" => "algorithms.md",
        "API" => "api.md",
        "Examples" => "examples.md",
        "Developer Notes" => "developer_notes.md",
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
