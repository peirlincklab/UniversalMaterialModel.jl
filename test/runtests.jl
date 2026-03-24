using Test, UniversalMaterialModel, Tensors

struct NeoHooke
    C₁₀::Float64
    D₁::Float64
end

function UniversalMaterialModel.Ψ(C, mp::NeoHooke)
    C₁₀ = mp.C₁₀
    D₁ = mp.D₁
    I₁ = tr(C)
    I₃ = det(C)
    return C₁₀ * (I₁ - 3) + inv(D₁) * (I₃ - 1)^2
end

function constitutive_driver(C, mp::NeoHooke)
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end


# import UniversalMaterialModel: cann_neuron
# function test_generated(C, mat::UniversalMaterial{Topology, Nfibers, N}) where {Topology, Nfibers, N}
#     T = eltype(C)
#     # Invariant indices actually referenced by this particular network.
#     needed = Set{Int}(Topology[i][1] for i in 1:N)
#     expr = Expr[]
#     # I₁ is also needed as an intermediate for I₂
#     !isempty(needed ∩ (1:2)) && push!(expr, :(I1 = tr(C)))
#     # I₂ = tr(C²) = dcontract(C,C) for symmetric C
#     2 ∈ needed && push!(expr, :(I2 = (I1 * I1 - dcontract(C, C)) / 2))
#     # I₃ in the list
#     3 ∈ needed && push!(expr, :(I3 = det(C)))
#     # fiber_contributions!(expr, fibers)
#     # build the strain energy
#     push!(expr, :(W = zero($T)))
#     for i in 1:N
#         kInv, kf0, kf1, kf2 = Topology[i]
#         inv_sym = Symbol(:I, kInv)
#         # Build reference expression: either a numeric literal or a fiber dot product.
#         ref     = UniversalMaterialModel._ref_expr(kInv)
#         ref_ex  = ref isa Expr ? ref : :($ref)
#         push!(expr, :(W += cann_neuron($(Val(kf0)), $(Val(kf1)), $(Val(kf2)), mat.weights[$i][1],
#                                         mat.weights[$i][2], mat.weights[$i][3], $inv_sym - $ref_ex)))
#     end
#     # return value
#     push!(expr, :(return W))
#     return Expr(:block, expr...)
# end

# Material parameters
E = 10.0
ν = 0.3
μ = E / (2(1 + ν))
λ = (E * ν) / ((1 + ν) * (1 - 2ν))
C₁₀ = μ/2.0
D₁  = 2.0 / (3.0 * μ + λ)
mp = NeoHooke(C₁₀, D₁)

# NeoHook model tab
# terms of the CaNN
terms = [(1.0,1.0,1.0,1.0,1.0,1.0,C₁₀),
         (3.0,1.0,2.0,1.0,2.0,1.0,inv(D₁))]

# should be exported
mat = UniversalMaterialModel.build_material(terms)

F = rand(Tensor{2, 3, Float64}) + one(Tensor{2, 3, Float64})
C = tdot(F)

Ψ(C, mat; fibers=())

Ψ(C, mp)

S, ∂S∂C = mat(C)
Sₑ,∂S∂Cₑ= constitutive_driver(C, mp)

# @time a = test_generated(C, mat)

# f1 = Vec{3}((1.0, 0.0, 0.0))
# f2 = Vec{3}((0.0, 1.0, 0.0))
# f3 = Vec{3}((0.0, 0.0, 1.0))

# get the material at this state
# S, ∂S∂C = mat(C)

# S, ∂S∂C = constitutive_driver(C, mp)

# NeoHook model
# terms = ()

# UniversalMaterial(terms)
