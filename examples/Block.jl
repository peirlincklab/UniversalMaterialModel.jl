using Ferrite, Tensors
using UniversalMaterialModel

function assemble_element!(ke, ge, cell, cv, fv, mat, ue, ΓN)
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)
    ndofs = getnbasefunctions(cv)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F) # F' ⋅ F
        # Compute stress and tangent
        S, ∂S∂C = mat(C)
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(I, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I)

        # Loop over test functions
        for i in 1:ndofs
            # Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            ge[i] += (∇δui ⊡ P) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function assemble_global!(K, g, dh, cv, fv, mat, u, ΓN)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    # start_assemble resets K and g
    assembler = start_assemble(K, g)

    # Loop over all cells in the grid
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs
        assemble_element!(ke, ge, cell, cv, fv, mat, ue, ΓN)
        assemble!(assembler, global_dofs, ke, ge)
    end
    return
end

function cauchy_stress(cell, cv, mat, ue)
    σ_avg = zero(SymmetricTensor{2,3})
    for qp in 1:getnquadpoints(cv)
        ∇u = function_gradient(cv, qp, ue)
        F  = one(∇u) + ∇u
        C  = tdot(F)            # Fᵀ·F
        S, _ = mat(C)
        J  = det(F)
        σ_avg += symmetric(F ⋅ S ⋅ F') / (J * getnquadpoints(cv))
    end
    return σ_avg
end

function max_principal_stress(dh, cv, mat, u)
    S_principal = zeros(getncells(dh.grid))
    for (i, cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        σ = cauchy_stress(cell, cv, mat, ue)
        S_principal[i] = maximum(eigvals(σ))
    end
    return S_principal
end

# Generate a grid
N = 16
L = 1.0
left = zero(Vec{3})
right = L * ones(Vec{3})
grid = generate_grid(Hexahedron, (N, N, N), left, right)

# Material parameters
E = 10.0
ν = 0.3
μ = E / (2(1 + ν))
λ = (E * ν) / ((1 + ν) * (1 - 2ν))

# NeoHook model tab
terms = [(1,1,1,1,1.0,1.0,μ/2),
         (3,1,2,1,1.0,1.0,λ/2)]
mat = UniversalMaterialModel.build_material(terms)

# Finite element base
ip = Lagrange{RefHexahedron, 1}()^3
qr = QuadratureRule{RefHexahedron}(2)
qr_facet = FacetQuadratureRule{RefHexahedron}(2)
cv = CellValues(qr, ip)
fv = FacetValues(qr_facet, ip)

# DofHandler
dh = DofHandler(grid)
add!(dh, :u, ip) # Add a displacement field
close!(dh)

function rotation(X, t)
    θ = pi / 2.0 # 90°
    x, y, z = X
    return Vec{3}((-t,L/2-y+(y-L/2)*cos(θ*t)-(z-L/2)*sin(θ*t),L/2-z+(y-L/2)*sin(θ*t)+(z-L/2)*cos(θ*t)))
end

dbcs = ConstraintHandler(dh)
# Add a homogeneous boundary condition on the "clamped" edge
add!(dbcs, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]))
add!(dbcs, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> rotation(x, t), [1, 2, 3]))
close!(dbcs)

# Neumann part of the boundary
ΓN = union(
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
    getfacetset(grid, "front"),
    getfacetset(grid, "back"),
)

# Pre-allocation of vectors for the solution and Newton increments
_ndofs = ndofs(dh)
un = zeros(_ndofs) # previous solution vector
u = zeros(_ndofs)
Δu = zeros(_ndofs)
ΔΔu = zeros(_ndofs)
apply!(un, dbcs)

# Create sparse matrix and residual vector
K = allocate_matrix(dh)
g = zeros(_ndofs)

# Perform Newton iterations
NEWTON_TOL = 1.0e-8
NEWTON_MAXITER = 100

let λᵢ=0; @time for λ in 0.0:0.01:0.6
    # Newton solve for current displacement step
    λᵢ += 1; newton_itr = -1
    # update the boundary conditions for the current load step
    Ferrite.update!(dbcs, λ)
    while true
        newton_itr += 1
        # Construct the current guess and enforce BCs at current λ
        u .= un .+ Δu
        apply!(u, dbcs)
        # Compute residual and tangent for current guess
        assemble_global!(K, g, dh, cv, fv, mat, u, ΓN)
        # Apply boundary conditions
        apply_zero!(K, g, dbcs)
        # Compute the residual norm and compare with tolerance
        normg = norm(g)
        if normg < NEWTON_TOL
            break
        elseif newton_itr > NEWTON_MAXITER
            error("Reached maximum Newton iterations, aborting at $(norm(g))")
        end

        # Compute Newton increment via direct solve
        ΔΔu .= K \ g

        apply_zero!(ΔΔu, dbcs)
        Δu .-= ΔΔu
    end
    println("Load step λ=$(round(λ; digits=2)) converged in $newton_itr iterations to $(norm(g))")
    # Commit converged solution and reset increment for next load step
    un .= u
    fill!(Δu, 0.0)
end;
end

# Save the solution
σ₁ = max_principal_stress(dh, cv, mat, u)
VTKGridFile("hyperelasticity", dh) do vtk
    write_solution(vtk, dh, u)
    write_cell_data(vtk, σ₁, "max_principal_stress")
end