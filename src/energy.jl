"""
    Ψ(C, mat::UniversalMaterial; fibers=())

Evaluate the `UniversalMaterial` at right Cauchy-Green deformation tensor `C=Fᵀ·F`.
The `fibers` keyword argument is required for anisotropic materials with fiber contributions
(Nfibers > 0) and should be `NTuple{Nfibers, Vec{3}}` of reference-configuration fiber unit vectors;
use `()` for isotropic materials.

@generated expression for `Ψ` is optimized to compute only the invariants actually referenced by the particular network
topology of this material, and to reuse common subexpressions (e.g. C·f₁) across multiple invariants when possible.
"""
@generated function Ψ(C, mat::UniversalMaterial{Topology, Nfibers, N, Compressible}; fibers::NTuple{Nfibers, Vec{3}}) where {Topology, Nfibers, N, Compressible}
    T = eltype(C)
    # Invariant indices actually referenced by this particular network.
    needed = Set{Int}(Topology[i][1] for i in 1:N)
    expr = Expr[]
    # I₁: needed directly, as intermediate for I₂, or as intermediate for Ī₁/Ī₂ (kInv 16/17)
    !isempty(needed ∩ Set([1, 2, 16, 17])) && push!(expr, :(I1 = tr(C)))
    # I₂: needed directly or as intermediate for Ī₂ (kInv 17)
    (2 ∈ needed || 17 ∈ needed) && push!(expr, :(I2 = (I1 * I1 - dcontract(C, C)) / 2))
    # I₃: needed directly, or as intermediate for isochoric invariants Ī₁/Ī₂
    (3 ∈ needed || !isempty(needed ∩ (16:17))) && push!(expr, :(I3 = det(C)))
    # Isochoric invariants for compressible decoupled formulation
    # Ī₁ = J^(-2/3) I₁ = I₃^(-1/3) I₁,   Ī₂ = J^(-4/3) I₂ = I₃^(-2/3) I₂
    if !isempty(needed ∩ (16:17))
        push!(expr, :(J23 = cbrt(I3)))           # I₃^(1/3) = J^(2/3), shared factor
        16 ∈ needed && push!(expr, :(I16 = I1 / J23))
        17 ∈ needed && push!(expr, :(I17 = I2 / (J23 * J23)))
    end
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
    push!(expr, :(W = zero($T)))
    for i in 1:N
        kInv, kf0, kf1, kf2 = Topology[i]
        inv_sym = Symbol(:I, kInv)
        # Build reference expression: either a numeric literal or a fiber dot product.
        ref    = ref_expr(kInv)
        ref_ex = ref isa Expr ? ref : :($ref)
        w0_ex  = :(mat.weights[$i][1])
        w1_ex  = :(mat.weights[$i][2])
        w2_ex  = :(mat.weights[$i][3])
        push!(expr, :(W += $(neuron_expr(Val(kf0), Val(kf1), Val(kf2), w0_ex, w1_ex, w2_ex, :($inv_sym - $ref_ex)))))
    end
    # return value
    push!(expr, :(return W))
    return Expr(:block, expr...)
end

# h₀ — zeroth layer
h0_expr(::Val{1}, x) = x                            # identity  →  x
h0_expr(::Val{2}, x) = :(max($x, zero($x)))         # Macaulay  →  max(x, 0)
h0_expr(::Val{3}, x) = :(abs($x))                   # absolute  →  abs(x)

# h₁ — first layer: (w·x)^p  (p is a compile-time integer literal)
h1_expr(::Val{p}, w, x) where {p} = :(($w * $x)^$p)

# h₂ — second layer
h2_expr(::Val{1}, w, x) = :($w * $x)
h2_expr(::Val{2}, w, x) = :(exp($w * $x) - one($x))
h2_expr(::Val{3}, w, x) = :(-log(one($x) - $w * $x))

# Full neuron expression: w2 · h2( w1 · h1( w0 · h0(x) ) )
function neuron_expr(::Val{k0}, ::Val{k1}, ::Val{k2}, w0, w1, w2, x) where {k0, k1, k2}
    inner = h0_expr(Val(k0), x)
    mid   = h1_expr(Val(k1), w0, inner)
    outer = h2_expr(Val(k2), w1, mid)
    return :($w2 * $outer)
end

# term to add to move invariant into reference configuration.
function ref_expr(k::Int)
    k == 1       && return 3.0
    k == 2       && return 3.0
    k == 3       && return 1.0
    k ∈ (4, 5)   && return 1.0
    k ∈ (6, 7)   && return :(fibers[1] ⋅ fibers[2])
    k ∈ (8, 9)   && return 1.0
    k ∈ (10, 11) && return :(fibers[1] ⋅ fibers[3])
    k ∈ (12, 13) && return :(fibers[2] ⋅ fibers[3])
    k ∈ (14, 15) && return 1.0
    k == 16      && return 3.0   # Ī₁ = 3 at reference (J=1, I₁=3)
    k == 17      && return 3.0   # Ī₂ = 3 at reference (J=1, I₂=3)
    error("Invariant index $k not supported (valid range: 1–17)")
end