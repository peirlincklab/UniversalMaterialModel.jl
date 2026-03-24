module UniversalMaterialModel

using Tensors

export UniversalMaterial, Ψ, evaluate, load_material

"""
    Each neuron computes:

      W_neuron(x) = w₂ · h₂( w₁ · h₁( w₀ · h₀(x) ) )

    where  x = Ĩₖ = Iₖ − Iₖ⁰  is the invariant shifted by its reference value
    (ensures W = 0 in the undeformed configuration).

    Layer roles:
      h₀  – input activation (controls domain)
      h₁  – polynomial power law
      h₂  – monotone output activation (guarantees energy growth conditions)
"""
# h₀ — zeroth layer
@inline h0(::Val{1}, x) = x                          # identity
@inline h0(::Val{2}, x) = max(x, zero(x))            # Macaulay bracket ⟨x⟩ (ReLU)
@inline h0(::Val{3}, x) = abs(x)                     # absolute value

# h₁ — first layer: (w·x)^p
@inline h1(::Val{p}, w, x) where {p} = (w * x)^p

# h₂ — second layer
@inline h2(::Val{1}, w, x) = w * x                   # linear
@inline h2(::Val{2}, w, x) = exp(w * x) - one(x)    # exp − 1   (unbounded growth)
@inline h2(::Val{3}, w, x) = -log(one(x) - w * x)   # −ln(1−wx) (singular barrier)

# Single CANN neuron — k values are Val{} so dispatch is resolved at compile time.
@inline function cann_neuron(::Val{k0}, ::Val{k1}, ::Val{k2}, w0, w1, w2, x) where {k0, k1, k2}
    w2 * h2(Val(k2), w1, h1(Val(k1), w0, h0(Val(k0), x)))
end

"""
    UniversalMaterial{Topology, Nfibers, N}

Incompressible CANN hyperelastic material.

Type parameters (compile-time — these drive the `@generated` specialisation):
  - `Topology` – `NTuple{N, NTuple{4,Int}}`: one `(kInv, k0, k1, k2)` tuple per neuron.
                  Encodes which invariant each neuron reads and which layer functions it uses.
  - `Nfibers`  – number of fiber families: 0 (isotropic), 1, 2, or 3.
  - `N`        – total number of neurons.

Runtime field:
  - `weights`  – `NTuple{N, NTuple{3,Float64}}`: one `(w0, w1, w2)` weight triple per neuron.

Construct via `load_material(filename)` — do not build directly.
"""
struct UniversalMaterial{Topology, Nfibers, N}
    weights::NTuple{N, NTuple{3, Float64}}
end

function Base.show(io::IO, mat::UniversalMaterial{Topology, Nfibers, N}) where {Topology, Nfibers, N}
    print(io, "UniversalMaterial(N=$N neurons, Nfibers=$Nfibers, topology=[")
    for i in 1:N
        kInv, k0, k1, k2 = Topology[i]
        i > 1 && print(io, ", ")
        print(io, "(I$kInv, h0=$k0, h1=$k1, h2=$k2)")
    end
    print(io, "])")
end

# Invariant definitions and reference values
#
# up to 3 fiber families (f₁, f₂, f₃):
#
#  Index  Formula                    Reference (C=I, unit fibers)
#  ─────  ─────────────────────────  ────────────────────────────
#    1    tr(C)                      3
#    2    (tr(C)² − tr(C²)) / 2     3
#    3    det(C)                     1
#    4    f₁ · C · f₁               1
#    5    f₁ · C² · f₁              1       [ = |C·f₁|² ]
#    6    f₁ · C · f₂               f₁·f₂
#    7    f₁ · C² · f₂              f₁·f₂   [ = (C·f₁)·(C·f₂) ]
#    8    f₂ · C · f₂               1
#    9    f₂ · C² · f₂              1       [ = |C·f₂|² ]
#   10    f₁ · C · f₃               f₁·f₃
#   11    f₁ · C² · f₃              f₁·f₃   [ = (C·f₁)·(C·f₃) ]
#   12    f₂ · C · f₃               f₂·f₃
#   13    f₂ · C² · f₃              f₂·f₃   [ = (C·f₂)·(C·f₃) ]
#   14    f₃ · C · f₃               1
#   15    f₃ · C² · f₃              1       [ = |C·f₃|² ]
#
# The nFibers–nInv mapping follows the Fortran subroutine:
#   Nfibers = 0 → nInv =  3  (I₁–I₃)
#   Nfibers = 1 → nInv =  5  (I₁–I₅)
#   Nfibers = 2 → nInv =  9  (I₁–I₉)
#   Nfibers = 3 → nInv = 15  (I₁–I₁₅)

# Code-generation helper (runs at @generated compile time, not at runtime).
# Returns either a Float64 literal or an Expr involving the runtime `fibers` argument.
function _ref_expr(k::Int)
    k == 1       && return 3.0
    k == 2       && return 3.0
    k == 3       && return 1.0
    k ∈ (4, 5)   && return 1.0
    k ∈ (6, 7)   && return :(fibers[1] ⋅ fibers[2])
    k ∈ (8, 9)   && return 1.0
    k ∈ (10, 11) && return :(fibers[1] ⋅ fibers[3])
    k ∈ (12, 13) && return :(fibers[2] ⋅ fibers[3])
    k ∈ (14, 15) && return 1.0
    error("Invariant index $k not supported (valid range: 1–15)")
end

# @generated strain energy  W(F)
#
# The generator runs once per distinct (Topology, Nfibers, N, T) combination
# and emits a flat, fully unrolled Expr with:
#   • no runtime branches on kInv / kf values
#   • only the invariants actually needed by the network computed
#   • fiber quantities computed only when required
@generated function Ψ(C, mat::UniversalMaterial{Topology, Nfibers, N}; fibers::NTuple{Nfibers, Vec{3}}) where {Topology, Nfibers, N}
    T = eltype(C)
    # Invariant indices actually referenced by this particular network.
    needed = Set{Int}(Topology[i][1] for i in 1:N)
    expr = Expr[]
    # I₁ is also needed as an intermediate for I₂
    !isempty(needed ∩ (1:2)) && push!(expr, :(I1 = tr(C)))
    # I₂ = tr(C²) = dcontract(C,C) for symmetric C
    2 ∈ needed && push!(expr, :(I2 = (I1 * I1 - dcontract(C, C)) / 2))
    # I₃ in the list
    3 ∈ needed && push!(expr, :(I3 = det(C)))
    # Fiber-1 quantities
    # f1 participates in: I₄, I₅ (fiber-1), I₆, I₇ (coupling 1-2), I₁₀, I₁₁ (coupling 1-3)
    need_f1 = Nfibers >= 1 && !isempty(needed ∩ Set([4, 5, 6, 7, 10, 11]))
    if need_f1
        push!(expr, :(f1  = fibers[1]))
        push!(expr, :(Cf1 = C ⋅ f1))          # C·f₁  (reused by I₄,I₅,I₇,I₁₁)
        4 ∈ needed && push!(expr, :(I4  = f1  ⋅ Cf1))   # f₁·C·f₁
        5 ∈ needed && push!(expr, :(I5  = Cf1 ⋅ Cf1))   # |C·f₁|²  = f₁·C²·f₁
    end
    # Fiber-2 quantities + coupling 1–2
    # f2 participates in: I₆, I₇ (coupling 1-2), I₈, I₉ (fiber-2), I₁₂, I₁₃ (coupling 2-3)
    need_f2 = Nfibers >= 2 && !isempty(needed ∩ Set([6, 7, 8, 9, 12, 13]))
    if need_f2
        push!(expr, :(f2  = fibers[2]))
        push!(expr, :(Cf2 = C ⋅ f2))          # C·f₂  (reused by I₆,I₇,I₈,I₉,I₁₃)
        6 ∈ needed && push!(expr, :(I6  = f1  ⋅ Cf2))   # f₁·C·f₂
        7 ∈ needed && push!(expr, :(I7  = Cf1 ⋅ Cf2))   # (C·f₁)·(C·f₂) = f₁·C²·f₂
        8 ∈ needed && push!(expr, :(I8  = f2  ⋅ Cf2))   # f₂·C·f₂
        9 ∈ needed && push!(expr, :(I9  = Cf2 ⋅ Cf2))   # |C·f₂|²  = f₂·C²·f₂
    end
    # Fiber-3 quantities + couplings 1–3 and 2–3
    # f3 participates in: I₁₀, I₁₁ (coupling 1-3), I₁₂, I₁₃ (coupling 2-3), I₁₄, I₁₅ (fiber-3)
    need_f3 = Nfibers >= 3 && !isempty(needed ∩ Set([10, 11, 12, 13, 14, 15]))
    if need_f3
        push!(expr, :(f3  = fibers[3]))
        push!(expr, :(Cf3 = C ⋅ f3))          # C·f₃  (reused by I₁₀–I₁₅)
        10 ∈ needed && push!(expr, :(I10 = f1  ⋅ Cf3))  # f₁·C·f₃
        11 ∈ needed && push!(expr, :(I11 = Cf1 ⋅ Cf3))  # (C·f₁)·(C·f₃) = f₁·C²·f₃
        12 ∈ needed && push!(expr, :(I12 = f2  ⋅ Cf3))  # f₂·C·f₃
        13 ∈ needed && push!(expr, :(I13 = Cf2 ⋅ Cf3))  # (C·f₂)·(C·f₃) = f₂·C²·f₃
        14 ∈ needed && push!(expr, :(I14 = f3  ⋅ Cf3))  # f₃·C·f₃
        15 ∈ needed && push!(expr, :(I15 = Cf3 ⋅ Cf3))  # |C·f₃|²  = f₃·C²·f₃
    end
    # Unrolled neuron contributions
    # Each iteration i is a compile-time constant; the loop is fully unrolled.
    # kf₀/kf₁/kf₂ are embedded as Val{} literals so layer-function dispatch
    # is resolved at compile time with zero runtime branching.
    push!(expr, :(W = zero($T)))
    for i in 1:N
        kInv, kf0, kf1, kf2 = Topology[i]
        inv_sym = Symbol(:I, kInv)
        # Build reference expression: either a numeric literal or a fiber dot product.
        ref     = _ref_expr(kInv)
        ref_ex  = ref isa Expr ? ref : :($ref)
        push!(expr, :(W += cann_neuron($(Val(kf0)), $(Val(kf1)), $(Val(kf2)), mat.weights[$i][1],
                                        mat.weights[$i][2], mat.weights[$i][3], $inv_sym - $ref_ex)))
    end
    # return value
    push!(expr, :(return W))
    return Expr(:block, expr...)
end
"""
    evaluate(mat, F, fibers) → (W, P, A)

Evaluate the `UniversalMaterial` at deformation gradient `F`.

# Arguments
- `mat`    – `UniversalMaterial` from `load_material`
- `F`      – deformation gradient `Tensor{2,3}` (e.g. from a Ferrite integration point)
- `fibers` – `NTuple{Nfibers, Vec{3}}` of reference-configuration fiber unit vectors;
             use `()` for isotropic materials

# Returns
- `W` – strain energy density                          (scalar)
- `P` – first Piola-Kirchhoff stress `∂W/∂F`          (`Tensor{2,3}`)
- `A` – material tangent             `∂²W/∂F∂F`       (`Tensor{4,3}`)

`P` and `A` are obtained by automatic differentiation of `W(F)` via
`Tensors.gradient` and `Tensors.hessian` (nested dual numbers).  No hand-coded
stress or tangent expressions are needed.

# Examples

Isotropic brain material (no fibers):
    mat     = load_material("brain-cnn.inp")
    F       = one(Tensor{2,3})
    W, P, A = evaluate(mat, F, ())

Two-fiber-family artery material:
    mat     = load_material("artery-cnn.inp"; material_name="MATADV")
    f1      = Vec{3}((1.0, 0.0, 0.0))
    f2      = Vec{3}((0.0, 1.0, 0.0))
    W, P, A = evaluate(mat, F, (f1, f2))
"""
# function evaluate(mat::UniversalMaterial{Topology, Nfibers, N},F::Tensor{2, 3, T},fibers::NTuple{Nfibers, Vec{3}},) where {Topology, Nfibers, N, T}
#     W_of_F = F_ -> Ψ(mat, F_, fibers)
#     P, W   = gradient(W_of_F, F, :all)    # ∇W, W(F)  — one forward pass
#     A      =  hessian(W_of_F, F)          # ∂²W/∂F²   — nested dual pass
#     return W, P, A
# end
# make it callable
function (mp::UniversalMaterial)(C::SymmetricTensor{2,3}; fibers=()) # fibres in kwargs
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp; fibers=fibers), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end


"""
    load_material(filename; material_name=nothing) → UniversalMaterial

Parse an Abaqus `.inp` file and construct a fully type-specialised `UniversalMaterial`.

The function reads `*PARAMETER TABLE, type="UNIVERSAL_TAB"` blocks, whose rows have
the format:
    kInv, kf0, kf1, kf2, w0, w1, w2

where:
  - `kInv`      – invariant index (1–15)
  - `kf0/1/2`   – layer function type for h₀, h₁, h₂
  - `w0/w1/w2`  – layer weights

# Arguments
- `filename`      – path to the `.inp` file
- `material_name` – (optional) name matching a `*MATERIAL, name=...` block;
                    when omitted, the first `UNIVERSAL_TAB` block in the file is used

# Notes
- `*INCLUDE` directives are silently skipped (the `UNIVERSAL_PARAM_TYPES.inc` schema
  file is not required)
- `Nfibers` is inferred from the maximum `kInv` used:
    - `kInv ≤ 3`  → 0 fibers (isotropic)
    - `kInv ≤ 5`  → 1 fiber family
    - `kInv ≤ 9`  → 2 fiber families
    - `kInv ≤ 15` → 3 fiber families
- The returned type has the full network topology encoded in type parameters.
  The first call to `evaluate` triggers compilation of a specialised, unrolled
  `strain_energy` function for that topology.
"""
function load_material(filename::String; material_name::Union{String, Nothing}=nothing)
    rows = _parse_universal_tab(filename, material_name)
    isempty(rows) && error(
        "No UNIVERSAL_TAB data found in \"$filename\"" *
        (material_name === nothing ? "" : " for material \"$material_name\""),
    )
    return build_material(rows)
end

"""
*PARAMETER TABLE, type="UNIVERSAL_TAB"
2,1,2,1,1.0,2.2725620E+00,2.2725594E-03
4,2,2,2,1.0,2.1151228E+01,8.1170170E-05
14,2,2,2,1.0,4.3718562E+00,3.1552851E-04
6,1,2,2,1.0,5.0801408E-01,4.8645619E-04
"""
# Each parsed row: (kInv, kf0, kf1, kf2, w0, w1, w2) stored as Float64 for uniform parsing.
const _Row = NTuple{7, Float64}

function _parse_universal_tab(filename::String,material_name::Union{String, Nothing}) :: Vector{_Row}
    rows   = _Row[]
    target = material_name !== nothing ? uppercase(strip(material_name)) : nothing

    # When no name filter is given we accept the very first UNIVERSAL_TAB found.
    in_target = (target === nothing)
    in_table  = false

    for raw in eachline(filename)
        line = strip(raw)
        isempty(line)           && continue
        startswith(line, "**") && continue      # Abaqus full-line comment

        if startswith(line, "*")
            # Any Abaqus keyword ends the current data block.
            in_table = false
            upper    = uppercase(line)

            if startswith(upper, "*MATERIAL")
                # Track whether we are inside the requested material section.
                if target !== nothing
                    m = match(r"name\s*=\s*([^\s,]+)"i, line)
                    in_target = (m !== nothing && uppercase(m[1]) == target)
                end

            elseif in_target && startswith(upper, "*PARAMETER TABLE")
                in_table = occursin("UNIVERSAL_TAB", upper)
            end

        elseif in_table
            # Data line: kInv,kf0,kf1,kf2,w0,w1,w2
            parts = split(line, ",")
            length(parts) < 7 && continue
            try
                row = ntuple(i -> parse(Float64, strip(parts[i])), 7)
                push!(rows, row)
            catch
                # silently skip malformed lines
            end
        end
    end

    return rows
end


function build_material(rows::Vector{_Row})
    N = length(rows)

    # The network topology — (kInv, kf0, kf1, kf2) per neuron — becomes a type parameter.
    # Julia allows NTuple{N, NTuple{4,Int}} as a type parameter (all isbits values).
    topology = ntuple(i -> ntuple(j -> Int(rows[i][j]), 4), N)
    weights  = ntuple(i -> (rows[i][5], rows[i][6], rows[i][7]), N)

    max_kinv = maximum(topology[i][1] for i in 1:N)
    Nfibers  = max_kinv ≤  3 ? 0 :
               max_kinv ≤  5 ? 1 :
               max_kinv ≤  9 ? 2 : 3

    # Explicitly specialise on the topology value — this is what drives @generated dispatch.
    return UniversalMaterial{topology, Nfibers, N}(weights)
end

end # module UniversalMaterialModel
