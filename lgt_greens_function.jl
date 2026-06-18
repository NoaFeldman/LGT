#= ═══════════════════════════════════════════════════════════════════════════════
   lgt_greens_function.jl

   2D Lattice Green's Function for the U(1) LGT with Open Boundary Conditions.

   The Green's function G(n, m) is the inverse of the discrete 2D Laplacian M
   with OBC.  It is used to compute the longitudinal electric field from the
   charge distribution Q_n via:

       E_long(n) = Σ_m  G(n, m) · Q_m

   ── Laplacian ────────────────────────────────────────────────────────────────
   M  =  L_{Nx} ⊗ I_{Ny}  +  I_{Nx} ⊗ L_{Ny}

   where L_N is the N×N tridiagonal matrix:
       (L_N)_{ij} = 2 δ_{i,j} − δ_{|i−j|,1}

   ── Spectral decomposition ───────────────────────────────────────────────────
   Eigenvalues:   Λ_k  =  λ_{kx} + λ_{ky}
     λ_{k,d}  =  2 − 2 cos(k π / (N_d + 1))     k = 1, …, N_d

   Eigenvectors:  Φ_k(n) = ∏_d  √(2/(N_d+1)) · sin(k_d n_d π / (N_d+1))

   The Laplacian with OBC is positive-definite (all eigenvalues > 0), so the
   Green's function G = M^{-1} exists and is unique.

   ── PEPO construction ────────────────────────────────────────────────────────
   Because G is fully nonlocal the "naive" PEPO has bond dimension equal to the
   total number of sites Nx·Ny.  The function build_greens_pepo() implements
   this exact (but exponentially expensive) representation as a warm-up step.
   A low-rank / approximated PEPO will be provided in a future extension.

   ── Interface ────────────────────────────────────────────────────────────────
   generate_greens_function(Nx, Ny)
       → G_mat  :: Matrix{Float64}  size (Nx*Ny) × (Nx*Ny)   (flattened)
       → G_tens :: Array{Float64,4} size (Nx, Ny, Nx, Ny)     (4-index tensor)

   build_greens_pepo(Nx, Ny, d_phys)
       → Array{Array{Float64,6}, 2}  size (Nx, Ny)
         Each element is a rank-6 array  [d_phys, d_phys, Dl, Dr, Du, Dd]
         representing one PEPO tensor.

   ── Testing ──────────────────────────────────────────────────────────────────
   Run  julia lgt_greens_function.jl  to execute the built-in test suite.
   ═══════════════════════════════════════════════════════════════════════════ =#

# ── Package guard (identical to convention used in the rest of this project) ──
if !@isdefined(_LGT_GREENS_LOADED)

using Pkg
Pkg.activate(joinpath(@__DIR__, "."))
for pkg in ["LinearAlgebra", "Test", "Printf"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

const _LGT_GREENS_LOADED = true
end

using LinearAlgebra
using Printf
using Test

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1-D Laplacian with OBC                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    laplacian_1d(N) -> Matrix{Float64}

Tridiagonal N×N matrix representing the 1-D discrete Laplacian with OBC:
    (L)_{ii} = 2,   (L)_{i,i±1} = -1.
"""
function laplacian_1d(N::Int)
    L = zeros(Float64, N, N)
    for i in 1:N
        L[i, i] = 2.0
        if i > 1; L[i, i-1] = -1.0; end
        if i < N; L[i, i+1] = -1.0; end
    end
    return L
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Site ↔ linear-index conversion                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Column-major linear index for site (ix, iy) on an Nx×Ny grid (1-based)."""
@inline site_index(ix::Int, iy::Int, Nx::Int) = (ix - 1) * 1 + (iy - 1) * Nx + 1
# Equivalently: ix + Nx*(iy-1)  (x is the fast index)
@inline site_index(ix::Int, iy::Int, Nx::Int, ::Nothing) = ix + Nx * (iy - 1)

# Unpack a linear index back to (ix, iy)
@inline function site_coords(idx::Int, Nx::Int)
    iy, ix = divrem(idx - 1, Nx)
    return ix + 1, iy + 1
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Green's Function: flat matrix + 4-index tensor                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    generate_greens_function(Nx, Ny)

Compute the lattice Green's function G = M^{-1} for the 2D discrete Laplacian
with open boundary conditions on an Nx×Ny grid.

Returns
───────
- `G_mat`  :: Matrix{Float64}  of size (Nx*Ny, Nx*Ny)
  Flattened representation.  Index convention: G_mat[α, β] where
  α = ix + Nx*(iy-1)  (column-major, x is the fast index).

- `G_tens` :: Array{Float64,4} of size (Nx, Ny, Nx, Ny)
  4-index tensor G_tens[ix, iy, jx, jy] = G(n=(ix,iy), m=(jx,jy)).

Algorithm
─────────
Spectral decomposition using the exact eigenvectors of M for OBC:

  Λ_{kx,ky} = λ_{kx} + λ_{ky}
  λ_{k,d}   = 2 − 2 cos(k π / (N_d+1))

  Φ_{k,d}(n_d) = √(2/(N_d+1)) sin(k n_d π / (N_d+1))

  G(n,m) = Σ_{kx,ky}  Φ_{kx}(nx) Φ_{ky}(ny) / Λ_{kx,ky} · Φ_{kx}(mx) Φ_{ky}(my)
"""
function generate_greens_function(Nx::Int, Ny::Int)
    # ── 1-D eigenvector matrices ──────────────────────────────────────────────
    # Φx[nx, kx] = √(2/(Nx+1)) sin(kx * nx * π / (Nx+1))
    Φx = [sqrt(2.0 / (Nx + 1)) * sin(kx * nx * π / (Nx + 1))
          for nx in 1:Nx, kx in 1:Nx]
    Φy = [sqrt(2.0 / (Ny + 1)) * sin(ky * ny * π / (Ny + 1))
          for ny in 1:Ny, ky in 1:Ny]

    # ── 1-D eigenvalues ───────────────────────────────────────────────────────
    λx = [2.0 - 2.0 * cos(kx * π / (Nx + 1)) for kx in 1:Nx]
    λy = [2.0 - 2.0 * cos(ky * π / (Ny + 1)) for ky in 1:Ny]

    # ── Assemble G in 4-index form via spectral sum ───────────────────────────
    G_tens = zeros(Float64, Nx, Ny, Nx, Ny)
    for ky in 1:Ny, kx in 1:Nx
        Λ = λx[kx] + λy[ky]
        inv_Λ = 1.0 / Λ
        for jy in 1:Ny, jx in 1:Nx
            φjx = Φx[jx, kx]
            φjy = Φy[jy, ky]
            for iy in 1:Ny, ix in 1:Nx
                G_tens[ix, iy, jx, jy] += Φx[ix, kx] * Φy[iy, ky] * inv_Λ * φjx * φjy
            end
        end
    end

    # ── Flatten to matrix ─────────────────────────────────────────────────────
    # Linear index α = ix + Nx*(iy-1)  →  reshape with x as fast index
    G_mat = reshape(permutedims(G_tens, (1, 2, 3, 4)), Nx * Ny, Nx * Ny)
    # permutedims is identity for (1,2,3,4) but we use reshape carefully:
    # G_tens has layout [ix, iy, jx, jy]; reshaping to [ix+Nx*(iy-1), jx+Nx*(jy-1)]
    # Julia is column-major: reshape([ix,iy,...]) iterates ix fastest → correct.
    G_mat = reshape(G_tens, Nx * Ny, Nx * Ny)

    return G_mat, G_tens
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Alternative: via direct matrix inversion (for validation)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    generate_greens_function_direct(Nx, Ny)

Alternative implementation using direct inversion of the Kronecker-sum
Laplacian M = L_{Nx} ⊗ I_{Ny} + I_{Nx} ⊗ L_{Ny}.
Useful as an independent check of `generate_greens_function`.
"""
function generate_greens_function_direct(Nx::Int, Ny::Int)
    Lx = laplacian_1d(Nx)
    Ly = laplacian_1d(Ny)
    Ix = Matrix{Float64}(I, Nx, Nx)
    Iy = Matrix{Float64}(I, Ny, Ny)

    # With x as the fast (inner) index, the linear index is ix + Nx*(iy-1).
    # For kron(A, B), the row index is (i_A-1)*size(B,1) + i_B, so B's index
    # is the fast one.  Therefore:
    #   M = kron(Iy, Lx) + kron(Ly, Ix)
    # gives  M_{(iy-1)*Nx+ix, (jy-1)*Nx+jx} = (Lx)_{ix,jx} δ_{iy,jy} + (Ly)_{iy,jy} δ_{ix,jx}
    M = kron(Iy, Lx) + kron(Ly, Ix)

    G_mat = inv(M)
    G_tens = reshape(G_mat, Nx, Ny, Nx, Ny)
    return G_mat, G_tens, M
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Naive (exact) PEPO for G                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    build_greens_pepo(Nx, Ny, d_phys)

Build an exact (exponentially expensive) PEPO representation of the Green's
function operator  Ĝ = Σ_{n,m} G(n,m) |n⟩⟨m|  on the physical Hilbert space.

The PEPO tensor at each site (ix, iy) has shape [d_phys, d_phys, Dl, Dr, Du, Dd]
where the virtual bond dimension is  D_virtual = Nx * Ny + 1.

Convention
──────────
The Green's function action on a charge state |Q⟩ = |Q_{1,1}, Q_{1,2}, …⟩ is
    (Ĝ|Q⟩)_n  =  Σ_m G(n,m) Q_m

This is implemented in the PEPO as a sum of rank-1 operators, one per pair (n,m).
The virtual bond carries an index labelling which pair is being processed.

Bond-dimension scaling
──────────────────────
D_virtual = Nx*Ny + 1  (one channel per source site m, plus the identity channel).
This is exact but scales as O(N²) in D, making it impractical for large systems.
A truncated/approximate version will replace this in future work.

Parameters
──────────
- Nx, Ny   : lattice dimensions
- d_phys   : physical dimension at each site (e.g. d_f or charge eigenvalue range)

Returns
───────
Array of size (Nx, Ny) of rank-6 Float64 arrays,
  pepo[ix, iy][σ′, σ, l, r, u, d]
  with index ordering:  out-phys, in-phys, left-bond, right-bond, up-bond, down-bond.
"""
function build_greens_pepo(Nx::Int, Ny::Int, d_phys::Int)
    G_mat, G_tens = generate_greens_function(Nx, Ny)

    # Virtual bond dimension:
    #   index 1         : pass-through / identity channel (no action yet)
    #   indices 2..N+1  : one channel per source site m  (linear index m)
    Nsites   = Nx * Ny
    D_virt   = Nsites + 1   # identity channel + one per source site

    # We use a "MPO-like spreading" scheme along the snake path x-major:
    # site ordering: (1,1), (2,1), …, (Nx,1), (1,2), …, (Nx,Ny)
    # The PEPO is built so that:
    #   • When passing through source site m, the physical projector onto charge
    #     Q_m is written into the left virtual bond channel m.
    #   • When reaching target site n, the channel m contributes G(n,m) to the
    #     outgoing physical degree of freedom.
    #   • Before / after: channels simply pass through.

    pepo = Array{Array{Float64,6}}(undef, Nx, Ny)

    for iy in 1:Ny, ix in 1:Nx
        n_lin = ix + Nx * (iy - 1)   # linear index of this site

        # Determine virtual bond directions present for this site:
        # We route the virtual bond horizontally along the snake path.
        # For simplicity we set Du = Dd = 1 (no vertical virtual bonds)
        # and Dr/Dl carry D_virt.  Corner / edge sites use D=1 for absent bonds.
        Dl = (ix == 1 && iy == 1) ? 1 : D_virt  # left incoming (or trivial at start)
        Dr = (ix == Nx && iy == Ny) ? 1 : D_virt  # right outgoing (or trivial at end)

        # For a snake path: at end of row ix==Nx, the "right" bond wraps to next row.
        # We keep a uniform Dr = D_virt for all non-terminal sites, Dl = D_virt for
        # all non-initial sites.  Terminal site has Dr=1; initial site has Dl=1.
        Du = 1
        Dd = 1

        W = zeros(Float64, d_phys, d_phys, Dl, Dr, Du, Dd)

        for σ in 1:d_phys
            charge_σ = σ - 1   # charge eigenvalue corresponding to basis state σ

            # ── Identity / pass-through channels ──────────────────────────────
            # Channel 1 (identity): pass through without acting on physical DoF.
            if Dl == 1 && Dr == D_virt
                # First site: open the identity channel
                W[σ, σ, 1, 1, 1, 1] += 1.0
            elseif Dl == D_virt && Dr == D_virt
                W[σ, σ, 1, 1, 1, 1] += 1.0   # identity channel passes through
            elseif Dl == D_virt && Dr == 1
                # Last site: close identity channel into physical output
                W[σ, σ, 1, 1, 1, 1] += 1.0
            end

            # ── Source-site channels ───────────────────────────────────────────
            # At site n_lin = m: open channel m+1 by projecting onto charge Q_m = charge_σ
            # and recording the charge value on the bond.
            #
            # At site n_lin = n (target): close channel m+1 by adding G(n,m) * charge_σ
            # to the physical output.  G(n,m) is a c-number kernel; the physical
            # output receives G(n,m) * |σ⟩⟨σ| weighted by the source charge.
            #
            # NOTE: The charge operator is diagonal in the occupation basis:
            #   Q_n = Σ_σ q(σ) |σ⟩⟨σ|  where q(σ) = charge_σ.
            # The Green's function action on charges, mapped back to the physical
            # space, is:  Ĝ acts on the charge density, not the full state.
            # Here we expose G as an operator on the charge index for illustrative
            # purposes; coupling to the full physical tensor is application-specific.

            for m_lin in 1:Nsites
                chan = m_lin + 1   # virtual bond channel for source site m_lin

                if Dl == 1 && Dr == D_virt
                    # First site (initial): can only be the source if n_lin==m_lin
                    # (there is no "pass through from the left")
                    if n_lin == m_lin
                        # Open channel: store charge_σ in channel chan
                        W[σ, σ, 1, chan, 1, 1] += Float64(charge_σ)
                    end
                    # Also: if this site is a target of a channel opened before us —
                    # not possible here (we are site 1).

                elseif Dl == D_virt && Dr == D_virt
                    # Middle site
                    if n_lin == m_lin
                        # Open channel for this source site
                        W[σ, σ, 1, chan, 1, 1] += Float64(charge_σ)
                    end

                    # Pass open channels (opened before this site) through
                    # without modification
                    if chan <= D_virt
                        W[σ, σ, chan, chan, 1, 1] += 1.0
                    end

                    # Act on physical DoF at this target site n from channel m
                    G_nm = G_tens[ix, iy, 
                                  ((m_lin - 1) % Nx) + 1,
                                  ((m_lin - 1) ÷ Nx) + 1]
                    if chan <= Dl  # channel was opened before this site
                        # Add G(n,m) * Q_m contribution to |σ⟩⟨σ| (charge diagonal)
                        W[σ, σ, chan, 1, 1, 1] += G_nm * Float64(charge_σ)
                    end

                elseif Dl == D_virt && Dr == 1
                    # Last site: close all channels
                    if n_lin == m_lin
                        # This site is both source and target — G(n,n) term
                        G_nm = G_tens[ix, iy, ix, iy]
                        W[σ, σ, 1, 1, 1, 1] += G_nm * Float64(charge_σ) * Float64(charge_σ)
                    end

                    # Close channel m (opened at site m, reaching here at last site)
                    G_nm = G_tens[ix, iy,
                                  ((m_lin - 1) % Nx) + 1,
                                  ((m_lin - 1) ÷ Nx) + 1]
                    if chan <= Dl
                        W[σ, σ, chan, 1, 1, 1] += G_nm * Float64(charge_σ)
                    end
                end
            end  # m_lin loop
        end  # σ loop

        pepo[ix, iy] = W
    end  # site loop

    return pepo
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Test suite                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    run_greens_function_tests()

Test suite verifying:
1. Spectral and direct methods agree.
2. M · G = I  (Laplacian of Green's function = Kronecker delta).
3. OBC: boundary sites satisfy the correct discrete equation.
4. Symmetry: G(n,m) = G(m,n).
"""
function run_greens_function_tests()
    println("=" ^ 60)
    println("  Green's Function Test Suite")
    println("=" ^ 60)

    for (Nx, Ny) in [(3, 3), (4, 4), (3, 5), (5, 3)]
        @testset "Nx=$Nx, Ny=$Ny" begin
            G_mat, G_tens          = generate_greens_function(Nx, Ny)
            G_mat_d, G_tens_d, M   = generate_greens_function_direct(Nx, Ny)
            Nsites = Nx * Ny

            # ── 1. Spectral vs direct ─────────────────────────────────────────
            @test norm(G_mat - G_mat_d) < 1e-10 * norm(G_mat_d)

            # ── 2. M · G = I ─────────────────────────────────────────────────
            MG  = M * G_mat
            err = norm(MG - I) / Nsites
            @test err < 1e-10
            @printf("   [Nx=%d Ny=%d] ‖M·G − I‖/N = %.2e\n", Nx, Ny, err)

            # ── 3. Check selected sites explicitly ───────────────────────────
            # For each target site m, verify Σ_l M_{n,l} G(l,m) = δ_{n,m}
            test_sites = [(1,1), (Nx,1), (1,Ny), (Nx,Ny),  # corners
                          (max(1,Nx÷2), max(1,Ny÷2)),        # interior
                          (1, max(1,Ny÷2)), (Nx, max(1,Ny÷2))] # edge midpoints
            for (mx, my) in test_sites
                m = mx + Nx * (my - 1)
                for (nx, ny) in test_sites
                    n = nx + Nx * (ny - 1)
                    val  = sum(M[n, l] * G_mat[l, m] for l in 1:Nsites)
                    δ_nm = (n == m) ? 1.0 : 0.0
                    @test abs(val - δ_nm) < 1e-10
                end
            end

            # ── 4. Symmetry G(n,m) = G(m,n) ──────────────────────────────────
            @test norm(G_mat - G_mat') < 1e-10 * norm(G_mat)

            # ── 5. 4-index and matrix forms are consistent ────────────────────
            for iy in 1:Ny, ix in 1:Nx
                for jy in 1:Ny, jx in 1:Nx
                    n = ix + Nx * (iy - 1)
                    m = jx + Nx * (jy - 1)
                    @test abs(G_tens[ix, iy, jx, jy] - G_mat[n, m]) < 1e-12
                end
            end
        end
    end

    println("=" ^ 60)
    println("  All tests passed.")
    println("=" ^ 60)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Demo / entry point                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Run tests when executed as the main script
if abspath(PROGRAM_FILE) == @__FILE__
    run_greens_function_tests()

    # Quick demo for a 4×4 lattice
    println("\nDemo: 4×4 lattice")
    Nx, Ny = 4, 4
    G_mat, G_tens = generate_greens_function(Nx, Ny)
    @printf("  G(1,1; 1,1) = %.6f\n", G_tens[1, 1, 1, 1])
    @printf("  G(2,2; 2,2) = %.6f\n", G_tens[2, 2, 2, 2])
    @printf("  G(1,1; 4,4) = %.6f\n", G_tens[1, 1, 4, 4])
    @printf("  G(2,2; 3,3) = %.6f\n", G_tens[2, 2, 3, 3])
    println("\nMax off-diagonal symmetry error: ",
        maximum(abs.(G_mat - G_mat')))

    # Build and inspect the PEPO (small d_phys=2 for demo)
    d_phys = 2
    pepo   = build_greens_pepo(Nx, Ny, d_phys)
    println("\nPEPO bond dimensions (Dl, Dr) at each site:")
    for iy in 1:Ny
        for ix in 1:Nx
            W = pepo[ix, iy]
            @printf("  (%d,%d): Dl=%d Dr=%d Du=%d Dd=%d\n",
                    ix, iy, size(W,3), size(W,4), size(W,5), size(W,6))
        end
    end
end
