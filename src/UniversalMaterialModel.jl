module UniversalMaterialModel

using Tensors

export UniversalMaterial, Ψ, load_material

"""
    UniversalMaterial{Topology, Nfibers, N, Compressible}

Universal material model.

Type parameters (compile-time — these drive the `@generated` specialisation):
  - `Topology`     – `NTuple{N, NTuple{4,Int}}`: one `(kInv, k0, k1, k2)` tuple per neuron.
                     Encodes which invariant each neuron reads and which layer functions it uses.
  - `Nfibers`      – number of fiber families: 0 (isotropic), 1, 2, or 3.
  - `N`            – total number of neurons.
  - `Compressible` – `Bool`: `true` when the network includes volumetric (kInv=3) or isochoric
                     (kInv=16 for Ī₁, kInv=17 for Ī₂) neurons; `false` for purely
                     isochoric/incompressible formulations.

Supported invariant indices:
  - 1–15 : standard structural invariants (I₁, I₂, I₃, fiber invariants)
  - 16   : Ī₁ = J^(-2/3) I₁  (isochoric first invariant, compressible split)
  - 17   : Ī₂ = J^(-4/3) I₂  (isochoric second invariant, compressible split)

Runtime field:
  - `weights`  – `NTuple{N, NTuple{3,Float64}}`: one `(w0, w1, w2)` weight triple per neuron.

Construct via `load_material(filename)`
```julia
mat = load_material("NeoHooke.inp")
```
or by passing tuples of topology and weights directly, example:
```julia
# Compressible Neo-Hookean:  Ψ = C₁₀(I₁−3) + (1/D₁)(J−1)²
C₁₀ = 2.0
D₁  = 0.1
terms = [(1.0,1.0,1.0,1.0,1.0,1.0,C₁₀),
         (3.0,1.0,2.0,1.0,1.0,1.0,inv(D₁))]
mat = UniversalMaterialModel.build_material(terms)
```
"""
struct UniversalMaterial{Topology, Nfibers, N, Compressible}
    weights::NTuple{N, NTuple{3, Float64}}
end

"""
    mat{UniversalMaterial}(C; fibers=()) → (S, ∂S∂C)

Evaluate the `UniversalMaterial` at right Cauchy-Green deformation tensor `C=Fᵀ·F`.

# Arguments
- `mat`    – `UniversalMaterial`
- `C`      – right Cauchy-Green deformation tensor `SymmetricTensor{2,3}` (e.g. from a Ferrite integration point)
- `fibers` – `NTuple{Nfibers, Vec{3}}` of reference-configuration fiber unit vectors;
              use `()` for isotropic materials

# Returns
- `S` – second Piola-Kirchhoff stress `∂Ψ/∂C`          (`SymmetricTensor{2,3}`)
- `∂S∂C` – material tangent           `∂²Ψ/∂C∂C`       (`SymmetricTensor{4,3}`)

`S` and `∂S∂C` are obtained by automatic differentiation of `Ψ(C)` via `Tensors.hessian` (nested dual numbers).

# Examples

Isotropic brain material (no fibers):
```julia
    mat     = load_material("brain-cnn.inp")
    F       = one(Tensor{2,3})
    C       = tdot(F)
    S, ∂S∂C = mat(C)
```
Two-fiber-family artery material:
```julia
    mat     = load_material("artery-cnn.inp"; material_name="MATADV")
    f1      = Vec{3}((1.0, 0.0, 0.0))
    f2      = Vec{3}((0.0, 1.0, 0.0))
    S, ∂S∂C = mat(C; fibers=(f1, f2))
```
"""
function (mp::UniversalMaterial)(C::SymmetricTensor{2,3}; fibers=()) # fibres in kwargs
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp; fibers=fibers), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end

# pretty print
function Base.show(io::IO, mat::UniversalMaterial{Topology, Nfibers, N, Compressible}) where {Topology, Nfibers, N, Compressible}
    print(io, "UniversalMaterial(N=$N neurons, Nfibers=$Nfibers, Compressible=$Compressible, topology=[")
    for i in 1:N
        kInv, k0, k1, k2 = Topology[i]
        i > 1 && print(io, ", ")
        print(io, "(I$kInv, h0=$k0, h1=$k1, h2=$k2)")
    end
    print(io, "])")
end

"""
    build_material(rows::Vector{Tuple{7}})

Builds a UniversalMaterial using a given vector of neurons, where every neuron is either
a `Tuple{7, Number}` or `NTuple{7, Float64}`.
"""
build_material(rows) = build_material([ntuple(i->Float64(row[i]),7) for row in rows])
function build_material(rows::Vector{NTuple{7, Float64}})
    N = length(rows)

    # The network topology — (kInv, kf0, kf1, kf2) per neuron — becomes a type parameter.
    topology = ntuple(i -> ntuple(j -> Int(rows[i][j]), 4), N)
    weights  = ntuple(i -> (rows[i][5], rows[i][6], rows[i][7]), N)

    kinvs = [topology[i][1] for i in 1:N]

    # kInv 16/17 are isochoric isotropic invariants — exclude them from the fiber count.
    fiber_kinvs = filter(k -> k ≤ 15, kinvs)
    max_kinv    = isempty(fiber_kinvs) ? 0 : maximum(fiber_kinvs)
    Nfibers     = max_kinv ≤  3 ? 0 :
                  max_kinv ≤  5 ? 1 :
                  max_kinv ≤  9 ? 2 : 3

    # Compressible if any neuron uses a volumetric (I₃) or isochoric (Ī₁, Ī₂) invariant.
    Compressible = any(k ∈ (3, 16, 17) for k in kinvs)

    return UniversalMaterial{topology, Nfibers, N, Compressible}(weights)
end

# include files
include("energy.jl")
include("io.jl")

end # module UniversalMaterialModel
