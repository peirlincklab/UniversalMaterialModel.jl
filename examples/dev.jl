using UniversalMaterialModel, Tensors

@generated function Ψ(C, mat::UniversalMaterial{Topology, Nfibers, N}) where {Topology, Nfibers, N}
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
    # build the strain energy
    push!(expr, :(W = zero($T)))
    for i in 1:N
        kInv, kf0, kf1, kf2 = Topology[i]
        inv_sym = Symbol(:I, kInv)
        # Build reference expression: either a numeric literal or a fiber dot product.
        ref     = UniversalMaterialModel.ref_expr(kInv)
        ref_ex  = ref isa Expr ? ref : :($ref)
        w1, w2, w3 = mat.weights[i]
        push!(expr, :(W += $(expression_neuron(Val(kf0), Val(kf1), Val(kf2), w1, w2, w3, :($inv_sym - $ref_ex)))))
    end
    # return value
    push!(expr, :(return W))
    return Expr(:block, expr...)
end

@inline h0_expr(::Val{1}, x) = :($x)                # identity
@inline h0_expr(::Val{2}, x) = :(max($x, zero($x)))  # Macaulay bracket ⟨x⟩ (ReLU)
@inline h0_expr(::Val{3}, x) = :(abs($x))           # absolute value
@inline h1_expr(::Val{p}, w, x) where {p} = :(($w * $x)^$p)
@inline h2_expr(::Val{1}, w, x) = :($w * $x)                  # linear
@inline h2_expr(::Val{2}, w, x) = :(exp($w * $x) - one($x))    # exp − 1   (unbounded growth)
@inline h2_expr(::Val{3}, w, x) = :(-log(one($x) - $w * $x))   # −ln(1−wx) (singular barrier)

function expression_neuron(::Val{k0}, ::Val{k1}, ::Val{k2}, w0, w1, w2, x) where {k0, k1, k2}
    :($w2 * $(h2_expr(Val(k2), w1, h1_expr(Val(k1), w0, h0_expr(Val(k0), x)))))
end

# deformation gradient and right Cauchy-Green tensor
F() = rand(Tensor{2, 3, Float64}) + one(Tensor{2, 3, Float64})
C = tdot(F())

# Material parameters for NeoHooke
C₁₀ = 2.0
D₁  = 0.1

# NeoHook model tab
terms = [(1.0,1.0,1.0,1.0,1.0,1.0,C₁₀),
        (3.0,1.0,2.0,1.0,1.0,1.0,inv(D₁))]
mat = UniversalMaterialModel.build_material(terms);

Ψ(C, mat)