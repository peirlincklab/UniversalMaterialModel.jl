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

struct MooneyRivlin
    C₁₀::Float64
    C₀₁::Float64
    D₁::Float64
end

function UniversalMaterialModel.Ψ(C, mp::MooneyRivlin)
    C₁₀ = mp.C₁₀
    C₀₁ = mp.C₀₁
    D₁ = mp.D₁
    I₁ = tr(C)
    I₂ = (I₁ * I₁ - dcontract(C, C)) / 2
    I₃ = det(C)
    return C₁₀ * (I₁ - 3) + C₀₁ * (I₂ - 3) + inv(D₁) * (I₃ - 1)^2
end

struct Yeon
    C₁₀::Float64
    C₂₀::Float64
    C₃₀::Float64
    D₁::Float64
    D₂::Float64
    D₃::Float64
end

function UniversalMaterialModel.Ψ(C, mp::Yeon)
    C₁₀ = mp.C₁₀
    C₂₀ = mp.C₂₀
    C₃₀ = mp.C₃₀
    D₁ = mp.D₁
    D₂ = mp.D₂
    D₃ = mp.D₃
    I₁ = tr(C)
    I₃ = det(C)
    return C₁₀ * (I₁ - 3) + C₂₀ * (I₁ - 3)^2 + C₃₀ * (I₁ - 3)^3 +
           inv(D₁) * (I₃ - 1)^2 + inv(D₂) * (I₃ - 1)^4 + inv(D₃) * (I₃ - 1)^6
end

struct Holzapfel
    C₁₀::Float64
    k₁::Float64
    k₂::Float64
    D ::Float64
    f₁::Vec{3,Float64}
    f₂::Vec{3,Float64}
end

function UniversalMaterialModel.Ψ(C, mp::Holzapfel)
    C₁₀ = mp.C₁₀
    k₁ = mp.k₁
    k₂ = mp.k₂
    D = mp.D
    I₁ = tr(C)
    I₃ = det(C)
    f₁ = mp.f₁
    f₂ = mp.f₂
    I₄₁₁ = f₁ ⋅ C ⋅ f₁
    I₄₂₂ = f₂ ⋅ C ⋅ f₂
    Tf₁ = k₁ / (2 * k₂) * (exp(k₂ * max((I₄₁₁ - 1)^2, 0)) - 1)
    Tf₂ = k₁ / (2 * k₂) * (exp(k₂ * max((I₄₂₂ - 1)^2, 0)) - 1)
    return C₁₀ * (I₁ - 3) + inv(D) * ((I₃^2 - 1) / 2 - log(I₃)) + Tf₁ + Tf₂
end

# evaluate any strain-energy function and its derivatives with respect to C
function constitutive_driver(C, mp)
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end

# deformation gradient and right Cauchy-Green tensor
F() = rand(Tensor{2, 3, Float64}) + one(Tensor{2, 3, Float64})
Cs = [tdot(F()) for _ in 1:10]

@testset "NeoHook model      " begin
    # Material parameters for NeoHooke
    C₁₀ = 2.0
    D₁  = 0.1
    mp = NeoHooke(C₁₀, D₁)

    # NeoHook model tab
    terms = [(1.0,1.0,1.0,1.0,1.0,1.0,C₁₀),
            (3.0,1.0,2.0,1.0,1.0,1.0,inv(D₁))]
    mat = UniversalMaterialModel.build_material(terms)

    for C in Cs
        # strain energies
        ψ = Ψ(C, mat; fibers=())
        ψₑ = Ψ(C, mp)
        @test ψ ≈ ψₑ
        # stress and tangent
        S, ∂S∂C = mat(C)
        Sₑ,∂S∂Cₑ= constitutive_driver(C, mp)
        @test all(S .≈ Sₑ)
        @test all(∂S∂C .≈ ∂S∂Cₑ)
    end
end

@testset "Mooney-Rivlin model" begin
    # Monly-Rivlin model tab
    C₁₀ = 1.0
    C₀₁ = 0.5
    D₁  = 0.1
    mp = MooneyRivlin(C₁₀, C₀₁, D₁)
    terms = [(1.0,1.0,1.0,1.0,1.0,1.0,C₁₀),
            (2.0,1.0,1.0,1.0,1.0,1.0,C₀₁),
            (3.0,1.0,2.0,1.0,1.0,1.0,inv(D₁))]
    mat = UniversalMaterialModel.build_material(terms)

    for C in Cs
        # strain energies
        ψ = Ψ(C, mat; fibers=())
        ψₑ = Ψ(C, mp)
        @test ψ ≈ ψₑ
        # stress and tangent
        S, ∂S∂C = mat(C)
        Sₑ,∂S∂Cₑ= constitutive_driver(C, mp)
        @test all(S .≈ Sₑ)
        @test all(∂S∂C .≈ ∂S∂Cₑ)
    end
end

@testset "Yeon model         " begin
    # Yeon model tab
    C₁₀ = 1.0
    C₂₀ = 0.5
    C₃₀ = 0.2
    D₁  = 0.1
    D₂  = 0.05
    D₃  = 0.01
    mp = Yeon(C₁₀, C₂₀, C₃₀, D₁, D₂, D₃)
    terms = [(1.0,1.0,1.0,1.0,1.0,1.0,C₁₀),
            (1.0,1.0,2.0,1.0,1.0,1.0,C₂₀),
            (1.0,1.0,3.0,1.0,1.0,1.0,C₃₀),
            (3.0,1.0,2.0,1.0,1.0,1.0,inv(D₁)),
            (3.0,1.0,4.0,1.0,1.0,1.0,inv(D₂)),
            (3.0,1.0,6.0,1.0,1.0,1.0,inv(D₃))]
    mat = UniversalMaterialModel.build_material(terms)

    for C in Cs
        # strain energies
        ψ = Ψ(C, mat; fibers=())
        ψₑ = Ψ(C, mp)
        @test ψ ≈ ψₑ
        # stress and tangent
        S, ∂S∂C = mat(C)
        Sₑ,∂S∂Cₑ= constitutive_driver(C, mp)
        @test all(S .≈ Sₑ)
        @test all(∂S∂C .≈ ∂S∂Cₑ)
    end
end

@testset "Holzapfel model    " begin
    # Holzapfel model tab
    C₁₀ = 1.0
    k₁  = 0.5
    k₂  = 2.0
    D   = 0.1
    f₁  = Vec(1.0, 0.0, 0.0)
    f₂  = Vec(0.0, 1.0, 0.0)
    mp = Holzapfel(C₁₀, k₁, k₂, D, f₁, f₂)
    terms = [(1.0,1.0,1.0,1.0,1.0,1.0,C₁₀),
            (4.0,2.0,2.0,2.0,1.0,k₂,k₁/2k₂),
            (8.0,2.0,2.0,2.0,1.0,k₂,k₁/2k₂),
            (3.0,1.0,1.0,1.0,1.0,1.0,inv(D)),
            (3.0,1.0,2.0,1.0,1.0,0.5,inv(D)),
            (3.0,1.0,1.0,3.0,1.0,-1.0,inv(D))]
    mat = UniversalMaterialModel.build_material(terms)

    for C in Cs
        # strain energies
        ψ = Ψ(C, mat; fibers=(f₁, f₂))
        ψₑ = Ψ(C, mp)
        @test ψ ≈ ψₑ
        # stress and tangent
        S, ∂S∂C = mat(C; fibers=(f₁, f₂))
        Sₑ,∂S∂Cₑ= constitutive_driver(C, mp)
        @test all(S .≈ Sₑ)
        @test all(∂S∂C .≈ ∂S∂Cₑ)
    end
end

struct IsochoricNeoHooke
    μ::Float64
    κ::Float64
end

# Decoupled compressible Neo-Hookean using isochoric/volumetric split:
#   Ψ = μ/2 (Ī₁ − 3) + κ (I₃ − 1)²,  Ī₁ = J^(-2/3) I₁ = I₃^(-1/3) I₁
function UniversalMaterialModel.Ψ(C, mp::IsochoricNeoHooke)
    I₁ = tr(C)
    I₃ = det(C)
    Ī₁ = I₃^(-1/3) * I₁
    return mp.μ/2 * (Ī₁ - 3) + mp.κ * (I₃ - 1)^2
end

struct IsochoricMooneyRivlin
    C₁₀::Float64
    C₀₁::Float64
    κ::Float64
end

# Decoupled compressible Mooney-Rivlin:
#   Ψ = C₁₀(Ī₁−3) + C₀₁(Ī₂−3) + κ(I₃−1)²
#   Ī₁ = I₃^(-1/3) I₁,  Ī₂ = I₃^(-2/3) I₂
function UniversalMaterialModel.Ψ(C, mp::IsochoricMooneyRivlin)
    I₁  = tr(C)
    I₂  = (I₁^2 - dcontract(C, C)) / 2
    I₃  = det(C)
    J23 = cbrt(I₃)
    Ī₁  = I₁ / J23
    Ī₂  = I₂ / (J23 * J23)
    return mp.C₁₀ * (Ī₁ - 3) + mp.C₀₁ * (Ī₂ - 3) + mp.κ * (I₃ - 1)^2
end

@testset "isochoric Neo-Hookean" begin
    μ  = 1.5
    κ  = 5.0
    mp = IsochoricNeoHooke(μ, κ)

    #  kInv=16 (Ī₁): μ/2·(Ī₁−3)  →  h₀=id, h₁=lin, h₂=lin, w₂=μ/2
    #  kInv=3  (I₃) : κ·(I₃−1)²  →  h₀=id, h₁=sq,  h₂=lin, w₂=κ
    terms = [(16.0, 1.0, 1.0, 1.0, 1.0, 1.0, μ/2),
             ( 3.0, 1.0, 2.0, 1.0, 1.0, 1.0, κ  )]
    mat = UniversalMaterialModel.build_material(terms)

    @test mat isa UniversalMaterial{<:Any, <:Any, <:Any, true}

    for C in Cs
        ψ = Ψ(C, mat; fibers=())
        @test ψ ≈ Ψ(C, mp)
        S, ∂S∂C = mat(C)
        Sₑ, ∂S∂Cₑ = constitutive_driver(C, mp)
        @test all(S .≈ Sₑ)
        @test all(∂S∂C .≈ ∂S∂Cₑ)
    end
end

@testset "isochoric Mooney-Rivlin" begin
    C₁₀ = 1.0
    C₀₁ = 0.4
    κ   = 8.0
    mp  = IsochoricMooneyRivlin(C₁₀, C₀₁, κ)

    #  kInv=16 (Ī₁): C₁₀·(Ī₁−3)
    #  kInv=17 (Ī₂): C₀₁·(Ī₂−3)
    #  kInv=3  (I₃) : κ·(I₃−1)²
    terms = [(16.0, 1.0, 1.0, 1.0, 1.0, 1.0, C₁₀),
             (17.0, 1.0, 1.0, 1.0, 1.0, 1.0, C₀₁),
             ( 3.0, 1.0, 2.0, 1.0, 1.0, 1.0, κ  )]
    mat = UniversalMaterialModel.build_material(terms)

    @test mat isa UniversalMaterial{<:Any, <:Any, <:Any, true}

    for C in Cs
        ψ = Ψ(C, mat; fibers=())
        @test ψ ≈ Ψ(C, mp)
        S, ∂S∂C = mat(C)
        Sₑ, ∂S∂Cₑ = constitutive_driver(C, mp)
        @test all(S .≈ Sₑ)
        @test all(∂S∂C .≈ ∂S∂Cₑ)
    end
end

@testset "Compressible flag     " begin
    # I₁-only network → incompressible
    mat_inc = UniversalMaterialModel.build_material([(1.0,1.0,1.0,1.0,1.0,1.0,1.0)])
    @test mat_inc isa UniversalMaterial{<:Any, <:Any, <:Any, false}

    # I₃ neuron → compressible
    mat_I3  = UniversalMaterialModel.build_material([(1.0,1.0,1.0,1.0,1.0,1.0,1.0),
                                                     (3.0,1.0,1.0,1.0,1.0,1.0,1.0)])
    @test mat_I3 isa UniversalMaterial{<:Any, <:Any, <:Any, true}

    # kInv=16 neuron → compressible
    mat_I16 = UniversalMaterialModel.build_material([(16.0,1.0,1.0,1.0,1.0,1.0,1.0)])
    @test mat_I16 isa UniversalMaterial{<:Any, <:Any, <:Any, true}

    # kInv=17 neuron → compressible
    mat_I17 = UniversalMaterialModel.build_material([(17.0,1.0,1.0,1.0,1.0,1.0,1.0)])
    @test mat_I17 isa UniversalMaterial{<:Any, <:Any, <:Any, true}
end

@testset "zero energy at ref    " begin
    C_ref = one(SymmetricTensor{2,3,Float64})
    f1    = Vec(1.0, 0.0, 0.0)
    f2    = Vec(0.0, 1.0, 0.0)

    C₁₀ = 2.0; D₁ = 0.1
    mat_NH = UniversalMaterialModel.build_material([(1.0,1.0,1.0,1.0,1.0,1.0,C₁₀),
                                                    (3.0,1.0,2.0,1.0,1.0,1.0,inv(D₁))])
    @test Ψ(C_ref, mat_NH; fibers=()) ≈ 0.0 atol=1e-12

    mat_MR = UniversalMaterialModel.build_material([(1.0,1.0,1.0,1.0,1.0,1.0,1.0),
                                                    (2.0,1.0,1.0,1.0,1.0,1.0,0.5),
                                                    (3.0,1.0,2.0,1.0,1.0,1.0,10.0)])
    @test Ψ(C_ref, mat_MR; fibers=()) ≈ 0.0 atol=1e-12

    mat_iso = UniversalMaterialModel.build_material([(16.0,1.0,1.0,1.0,1.0,1.0,1.0),
                                                     (17.0,1.0,1.0,1.0,1.0,1.0,0.5),
                                                     ( 3.0,1.0,2.0,1.0,1.0,1.0,5.0)])
    @test Ψ(C_ref, mat_iso; fibers=()) ≈ 0.0 atol=1e-12

    k₁ = 0.5; k₂ = 2.0; D = 0.1
    mat_Hz = UniversalMaterialModel.build_material([(1.0,1.0,1.0,1.0,1.0,1.0,2.0),
                                                    (4.0,2.0,2.0,2.0,1.0,k₂,k₁/2k₂),
                                                    (8.0,2.0,2.0,2.0,1.0,k₂,k₁/2k₂),
                                                    (3.0,1.0,1.0,1.0,1.0,1.0,inv(D)),
                                                    (3.0,1.0,2.0,1.0,1.0,0.5,inv(D)),
                                                    (3.0,1.0,1.0,3.0,1.0,-1.0,inv(D))])
    @test Ψ(C_ref, mat_Hz; fibers=(f1, f2)) ≈ 0.0 atol=1e-12
end

@testset "h₀=abs activation     " begin
    # h₀=3 (abs) is equivalent to h₀=1 (identity) when the input is non-negative.
    # I₁ − 3 ≥ 0 always for physically meaningful C, so both activations agree.
    terms_id  = [(1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.7)]  # h₀=identity
    terms_abs = [(1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 0.7)]  # h₀=abs
    mat_id    = UniversalMaterialModel.build_material(terms_id)
    mat_abs   = UniversalMaterialModel.build_material(terms_abs)

    for C in Cs
        @test Ψ(C, mat_id; fibers=()) ≈ Ψ(C, mat_abs; fibers=())
    end
end

@testset "loading inp table     " begin
    mat = load_material(joinpath(dirname(@__FILE__), "material.inp"))
    @test mat !== nothing
    # 4 neurons, 3 fiber families, kInv ∈ {2,4,6,14} → no volumetric term
    @test mat isa UniversalMaterial{<:Any, 3, 4, false}
    # values are non-trivial at a generic deformation
    C   = first(Cs)
    f1  = Vec(1.0, 0.0, 0.0)
    f2  = Vec(0.0, 1.0, 0.0)
    f3  = Vec(0.0, 0.0, 1.0)
    S, ∂S∂C = mat(C; fibers=(f1, f2, f3))
    @test !all(iszero, S)
    @test !all(iszero, ∂S∂C)
end

@testset "Base.show             " begin
    mat = UniversalMaterialModel.build_material([(1.0,1.0,1.0,1.0,1.0,1.0,1.0),
                                                 (3.0,1.0,2.0,1.0,1.0,1.0,10.0)])
    str = sprint(show, mat)
    @test occursin("Compressible=true",  str)
    @test occursin("N=2",                str)
    @test occursin("Nfibers=0",          str)
end