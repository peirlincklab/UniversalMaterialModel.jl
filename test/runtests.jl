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

@testset "loading inp table  " begin
    mat = load_material(joinpath(dirname(@__FILE__), "material.inp"))
    @test mat !== nothing
end