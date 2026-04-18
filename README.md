[![Test](https://github.com/marinlauber/UniversalMaterialModel.jl/actions/workflows/test.yml/badge.svg)](https://github.com/marinlauber/UniversalMaterialModel.jl/actions/workflows/test.yml)
[![codecov.io](https://codecov.io/github/marinlauber/UniversalMaterialModel.jl/coverage.svg?branch=master)](https://codecov.io/github/marinlauber/UniversalMaterialModel.jl?branch=master)

# UniversalMaterialModel.jl

A universal material model for [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)-based finite element code.

The implementation follows the ideas presented in the article:
[**"A universal material model subroutine for soft matter systems", M. Peirlinck, J.A. Hurtado, M.K. Rausch, A. Buganza Tepole, E. Kuhl, Engineering with Computers, 2024**](https://doi.org/10.1007/s00366-024-02031-w).

## Installation

The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia
pkg> add UniversalMaterialModel
```
Or, equivalently, via the Pkg API:

```julia
julia> import Pkg; Pkg.add("UniversalMaterialModel")
```

## How to use

You can create a `UniversalMaterial` either by loading an `.inp` Abaqus material table

```julia
using UniversalMaterialModel
# load the data from the inp file
material = load_material("NoeHooke.inp")
```
or by passing a `Vector` of `NTuple{7, Float64}` representing each neuron of the material model
```julia
using UniversalMaterialModel
# NeoHook model tab (Inv, h₀, h₁, h₂, w₀, w₁, w₂)
terms = [(1, 1, 1, 1, 1.0, 1.0, μ/2),
         (3, 1, 2, 1, 1.0, 1.0, λ/2)]
# build the material model
material = UniversalMaterialModel.build_material(terms)
```
The `UniversalMaterial` can then be called within your assembly procedure
```julia
function assemble_element!(...)
    # some code
    # ...
    for qp in 1:num_quad_points
        # some code
        # ...
        C = F' ⋅ F
        # Compute stress and tangent
        S, ∂S∂C = material(C) # callable
        P = F ⋅ S
        I = one(S)
        ∂P∂F = I ⊗ S + 2 * F ⋅ ∂S∂C ⊡ F' ⊗ I
        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            ge[i] += (∇δui ⊡ P) * dΩ
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += (∇δui ⊡ ∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end
end
```

If your model contains fibers, the call becomes
```julia
# fiber direction at integration point
f1 = Vec{3}((1.0, 0.0, 0.0))
f2 = Vec{3}((0.0, 1.0, 0.0))
# evaluate
S, ∂S∂C = material(C; fibers=(f1, f2))
```
where `fibers` is a tuple containing the fiber directions at the integration point, with a maximum of three fiber directions (`f₁`, `f₂`, `f₃`).

### Invariant definitions and reference values

| Index | Formula                  |  Reference (unit fibers) |
|-------|--------------------------|--------------------------|
|   1   | `tr(C)`                  | `3`                      |
|   2   | `(tr(C)² − tr(C²)) / 2`  | `3`                      |
|   3   | `det(C)`                 | `1`                      |
|   4   | `f₁ · C · f₁`            | `1`                      |
|   5   | `f₁ · C² · f₁`           | `1`                      |
|   6   | `f₁ · C · f₂`            | `f₁·f₂`                  |
|   7   | `f₁ · C² · f₂`           | `f₁·f₂`                  |
|   8   | `f₂ · C · f₂`            | `1`                      |
|   9   | `f₂ · C² · f₂`           | `1`                      |
|  10   | `f₁ · C · f₃`            | `f₁·f₃`                  |
|  11   | `f₁ · C² · f₃`           | `f₁·f₃`                  |
|  12   | `f₂ · C · f₃`            | `f₂·f₃`                  |
|  13   | `f₂ · C² · f₃`           | `f₂·f₃`                  |
|  14   | `f₃ · C · f₃`            | `1`                      |
|  15   | `f₃ · C² · f₃`           | `1`                      |

## How to Cite

If you use UniversalMaterialModel.jl for research and publication, please cite the following article.

```bibtex
@article{Peirlinck2024,
  title = {A universal material model subroutine for soft matter systems},
  volume = {41},
  ISSN = {1435-5663},
  url = {http://dx.doi.org/10.1007/s00366-024-02031-w},
  DOI = {10.1007/s00366-024-02031-w},
  number = {2},
  journal = {Engineering with Computers},
  publisher = {Springer Science and Business Media LLC},
  author = {Peirlinck,  Mathias and Hurtado,  Juan A. and Rausch,  Manuel K. and Tepole,  Adrián Buganza and Kuhl,  Ellen},
  year = {2024},
  month = sep,
  pages = {905–927}
}
```

### Issues and Support

Please use the GitHub issue tracker to report any issues.

### License

UniversalMaterialModel.jl is released under the MIT License. See the [LICENSE](LICENSE) file for details.