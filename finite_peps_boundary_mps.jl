#= ═══════════════════════════════════════════════════════════════════════════════
   finite_peps_boundary_mps.jl  —  "v4" finite boundary-MPS environment

   Stage 1 of the boundary-MPS full update: build the exact (χ-truncated) 2D
   environment of a horizontal/vertical bond by contracting the double-layer
   PEPS network row-by-row (top + bottom boundary MPS) and then sweeping the
   bond row.

   This replaces the diagonal Vidal-weight environment of the simple update
   (which froze v3 in the classical vacuum) with the real contracted
   environment that "sees" multi-bond flux-string correlations.

   Conventions
   ───────────
   • Symmetric gauge: every site absorbs sqrt(λ) on each existing leg, so the
     bare network of absorbed tensors reconstructs |ψ⟩ (each bond weight λ is
     split sqrt·sqrt across its two sites).
   • Double-layer tensor of a site has merged legs  X² ≡ (X_ket, X_bra),
     dim = D_X².  A boundary MPS tensor has legs (left_vbond, phys=D², right_vbond).
   • Boundary MPS for "top env at row r" carries phys legs = the up-double-legs
     (U²) of row r, i.e. it is the contraction of every row above r.

   Requires: finite_peps_gauge_invariant.jl (GaugePEPS, nx, ny),
             finite_peps_full_update.jl     (_dl_traced, _dl_open).
   ═══════════════════════════════════════════════════════════════════════════ =#

using LinearAlgebra
using TensorOperations

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Symmetric (sqrt) weight absorption                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    sqrt_absorbed_site(gp, ix, iy) → Array{ComplexF64,5}

Site tensor with sqrt(λ) absorbed on every existing leg, so that the bare
contraction of all such tensors reproduces |ψ⟩.  Boundary (length-1) weights
are 1.
"""
function sqrt_absorbed_site(gp::GaugePEPS, ix::Int, iy::Int)
    peps = gp.peps
    T = copy(peps.tensors[ix, iy])
    dp = size(T, 1)
    sλ_l = (ix > 1)        ? sqrt.(peps.λh[ix-1, iy]) : ones(1)
    sλ_r = (ix < nx(gp))   ? sqrt.(peps.λh[ix,   iy]) : ones(1)
    sλ_u = (iy < ny(gp))   ? sqrt.(peps.λv[ix,   iy]) : ones(1)
    sλ_d = (iy > 1)        ? sqrt.(peps.λv[ix, iy-1]) : ones(1)
    @inbounds for d in axes(T,5), u in axes(T,4), r in axes(T,3), l in axes(T,2), p in 1:dp
        T[p,l,r,u,d] *= sλ_l[l] * sλ_r[r] * sλ_u[u] * sλ_d[d]
    end
    return T
end

"""Traced double-layer tensor (Dl²,Dr²,Du²,Dd²) of the sqrt-absorbed site."""
dl_traced_site(gp::GaugePEPS, ix::Int, iy::Int) =
    _dl_traced(sqrt_absorbed_site(gp, ix, iy))

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MPS compression (truncate virtual bonds to χ)                           ║
# ║  MPS tensor leg order: (left, phys, right).                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    compress_mps!(mps, χ) → mps

Two-sweep MPS compression: left-to-right QR canonicalisation followed by a
right-to-left SVD sweep truncating every virtual bond to ≤ χ.  In place.
"""
function compress_mps!(mps::Vector{Array{ComplexF64,3}}, χ::Int)
    n = length(mps)
    n == 1 && return mps

    # ── left → right: QR, push R into the next site ──────────────────────
    for i in 1:n-1
        Dl, dp, Dr = size(mps[i])
        M = reshape(mps[i], Dl*dp, Dr)
        F = qr(M)
        Q = Matrix(F.Q); R = Matrix(F.R)
        k = size(Q, 2)
        mps[i] = reshape(Q, Dl, dp, k)
        Dl2, dp2, Dr2 = size(mps[i+1])
        nxt = reshape(mps[i+1], Dl2, dp2*Dr2)
        nxt = R * nxt
        mps[i+1] = reshape(nxt, k, dp2, Dr2)
    end

    # ── right → left: SVD, truncate to χ, push U·S into the previous site ─
    for i in n:-1:2
        Dl, dp, Dr = size(mps[i])
        M = reshape(mps[i], Dl, dp*Dr)
        F = svd(M)
        k = min(χ, count(>(1e-14), F.S), length(F.S))
        k = max(k, 1)
        U  = F.U[:, 1:k]
        S  = F.S[1:k]
        Vt = F.Vt[1:k, :]
        mps[i] = reshape(Vt, k, dp, Dr)
        Dl2, dp2, Dr2 = size(mps[i-1])
        prev = reshape(mps[i-1], Dl2*dp2, Dr2)
        prev = prev * (U * Diagonal(S))
        mps[i-1] = reshape(prev, Dl2, dp2, k)
    end
    return mps
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Boundary MPS construction                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Row `iy` traced double tensors, one per column (legs Dl²,Dr²,Du²,Dd²)."""
_row_dl(gp::GaugePEPS, iy::Int) = [dl_traced_site(gp, ix, iy) for ix in 1:nx(gp)]

"""Reshape a row of double tensors into a boundary MPS with phys = Dd² (down)."""
function _row_as_mps_down(row::Vector{Array{ComplexF64,4}})
    # E legs (L²,R²,U²,D²); U² == 1 for the top row → phys = D².
    return [reshape(permutedims(E, (1,4,2,3)), size(E,1), size(E,4), size(E,2)*size(E,3))
            for E in row]   # (L², D², R²·U²) ; U²=1 so == (L², D², R²)
end

"""
    absorb_row_from_top(mps, row) → new mps

Contract a lower row into the boundary MPS coming from above.  The row's up
leg (U²) contracts with the MPS phys leg; the row's down leg (D²) becomes the
new phys leg; virtual bonds combine.
"""
function absorb_row_from_top(mps::Vector{Array{ComplexF64,3}},
                             row::Vector{Array{ComplexF64,4}})
    n = length(mps)
    out = Vector{Array{ComplexF64,3}}(undef, n)
    for ix in 1:n
        a, u, b = size(mps[ix])                 # (left, phys=U², right)
        E = row[ix]                             # (L², R², U², D²)
        L2, R2, U2, D2 = size(E)
        @assert U2 == u "top-absorb phys mismatch at col $ix: $U2 vs $u"
        newM = zeros(ComplexF64, a*L2, D2, b*R2)
        @inbounds for rr in 1:R2, dd in 1:D2, ll in 1:L2, bb in 1:b, aa in 1:a
            s = zero(ComplexF64)
            for uu in 1:U2
                s += mps[ix][aa, uu, bb] * E[ll, rr, uu, dd]
            end
            newM[(aa-1)*L2+ll, dd, (bb-1)*R2+rr] = s
        end
        out[ix] = newM
    end
    return out
end

"""
    absorb_row_from_bottom(mps, row) → new mps

Mirror of `absorb_row_from_top` for the bottom boundary MPS: the row's down
leg (D²) contracts with the MPS phys; the row's up leg (U²) becomes new phys.
"""
function absorb_row_from_bottom(mps::Vector{Array{ComplexF64,3}},
                                row::Vector{Array{ComplexF64,4}})
    n = length(mps)
    out = Vector{Array{ComplexF64,3}}(undef, n)
    for ix in 1:n
        a, d, b = size(mps[ix])                 # (left, phys=D², right)
        E = row[ix]                             # (L², R², U², D²)
        L2, R2, U2, D2 = size(E)
        @assert D2 == d "bottom-absorb phys mismatch at col $ix: $D2 vs $d"
        newM = zeros(ComplexF64, a*L2, U2, b*R2)
        @inbounds for rr in 1:R2, uu in 1:U2, ll in 1:L2, bb in 1:b, aa in 1:a
            s = zero(ComplexF64)
            for dd in 1:D2
                s += mps[ix][aa, dd, bb] * E[ll, rr, uu, dd]
            end
            newM[(aa-1)*L2+ll, uu, (bb-1)*R2+rr] = s
        end
        out[ix] = newM
    end
    return out
end

"""
    boundary_top(gp, stop_row; χ) → mps with phys = up-legs (U²) of row `stop_row`

Contraction of all rows above `stop_row` (rows stop_row+1 … ny).
"""
function boundary_top(gp::GaugePEPS, stop_row::Int; χ::Int)
    nyv = ny(gp)
    # No rows above the top row → trivial environment (phys = U² = 1).
    stop_row >= nyv && return [reshape(ComplexF64[1.0], 1, 1, 1) for _ in 1:nx(gp)]
    mps = _row_as_mps_down(_row_dl(gp, nyv))          # phys = up-leg of row ny-1
    for r in (nyv-1):-1:(stop_row+1)
        mps = absorb_row_from_top(mps, _row_dl(gp, r))
        compress_mps!(mps, χ)
    end
    return mps
end

"""Reshape a row of double tensors into a boundary MPS with phys = Du² (up)."""
function _row_as_mps_up(row::Vector{Array{ComplexF64,4}})
    # bottom row has D² == 1 → phys = U².
    return [reshape(permutedims(E, (1,3,2,4)), size(E,1), size(E,3), size(E,2)*size(E,4))
            for E in row]   # (L², U², R²·D²) ; D²=1 so == (L², U², R²)
end

"""
    boundary_bottom(gp, stop_row; χ) → mps with phys = down-legs (D²) of row `stop_row`

Contraction of all rows below `stop_row` (rows 1 … stop_row-1).
"""
function boundary_bottom(gp::GaugePEPS, stop_row::Int; χ::Int)
    # No rows below the bottom row → trivial environment (phys = D² = 1).
    stop_row <= 1 && return [reshape(ComplexF64[1.0], 1, 1, 1) for _ in 1:nx(gp)]
    mps = _row_as_mps_up(_row_dl(gp, 1))              # phys = down-leg of row 2
    for r in 2:(stop_row-1)
        mps = absorb_row_from_bottom(mps, _row_dl(gp, r))
        compress_mps!(mps, χ)
    end
    return mps
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Full-lattice norm  ⟨ψ|ψ⟩  (self-test / normalisation)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    peps_norm2(gp; χ) → ComplexF64

Contract the entire double-layer network via top boundary MPS to get ⟨ψ|ψ⟩.
Should be real and positive for a valid state.
"""
function peps_norm2(gp::GaugePEPS; χ::Int=64)
    nyv = ny(gp)
    mps = _row_as_mps_down(_row_dl(gp, nyv))
    for r in (nyv-1):-1:1
        mps = absorb_row_from_top(mps, _row_dl(gp, r))
        compress_mps!(mps, χ)
    end
    # Bottom row reached: its phys legs are D² with dim 1 (row 1 has no down
    # link), and virtual bonds close on dim-1 boundaries → scalar.
    val = ones(ComplexF64, 1)
    acc = reshape(val, 1, 1)
    for ix in 1:length(mps)
        a, p, b = size(mps[ix])
        @assert p == 1 "norm: bottom phys dim ≠ 1 (got $p) at col $ix"
        acc = acc * reshape(mps[ix], a, b)
    end
    return acc[1, 1]
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Stage-1 self-test                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    bmps_selftest(; nx=3, ny=4, dg=1, g=1.0, m=0.25, χ=64, nsteps=20)

Quick consistency checks for the boundary-MPS machinery:
  1. ⟨ψ|ψ⟩ is real & positive.
  2. boundary_top / boundary_bottom have the expected phys dims.
  3. gauge invariance is preserved after a few v3 ITE steps (sanity hook).
Prints PASS/FAIL lines; does not mutate global state.
"""
function bmps_selftest(; nxv::Int=3, nyv::Int=4, dg::Int=1,
                        g::Float64=1.0, t_hop::Float64=1.0, m::Float64=0.25,
                        χ::Int=64, nsteps::Int=20, noise::Float64=0.1)
    gch = zeros(Int, nxv, nyv)
    for iy in 1:nyv, ix in 1:nxv
        isodd(ix + iy) && (gch[ix, iy] = -1)
    end
    gp = init_gauge_peps(nxv, nyv, dg, gch; noise=noise)
    # a few simple-update steps to give the bonds nontrivial structure
    for _ in 1:nsteps
        trotter_step_gauge!(gp; gcoup=g, t_hop=t_hop, m=m, τ=0.05, D_max=6)
    end

    println("─── boundary-MPS Stage-1 self-test ───")
    nrm = peps_norm2(gp; χ=χ)
    @printf("  ⟨ψ|ψ⟩ = %.6e + %.2e i   (imag should be ~0, real > 0)\n",
            real(nrm), imag(nrm))
    ok_norm = real(nrm) > 0 && abs(imag(nrm)) < 1e-8 * abs(real(nrm)) + 1e-12
    println(ok_norm ? "  PASS: norm real & positive" : "  FAIL: norm not real-positive")

    for iy in 2:nyv-1
        mt = boundary_top(gp, iy; χ=χ)
        mb = boundary_bottom(gp, iy; χ=χ)
        # phys dim of top env at row iy = (Du of site)² ; just report
        @printf("  row %d: top-mps phys dims = %s   bottom-mps phys dims = %s\n",
                iy, string([size(M,2) for M in mt]), string([size(M,2) for M in mb]))
    end

    viol = forbidden_norm(gp)
    @printf("  gauge_viol after %d steps = %.2e\n", nsteps, viol)
    println(viol < 1e-10 ? "  PASS: gauge invariance preserved" :
                           "  FAIL: gauge invariance broken")
    return ok_norm && viol < 1e-10
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Stage 2a: bond environment via QR reduction + boundary MPS              ║
# ║                                                                          ║
# ║  For a horizontal bond A=(ix,iy), B=(ix+1,iy):                          ║
# ║    • QR-reduce A over (outer = l,u,d | inner = p,bond) → Q_A · R_A      ║
# ║      and B over (inner = p,bond | outer = r,u,d) → R_B · Q_B            ║
# ║    • absorb Q_A, Q_B (ket & bra) into the boundary-MPS row contraction   ║
# ║      so the environment is a small tensor  env[aA,bB,aA',bB'].          ║
# ║    • ⟨ψ|ψ⟩ = Σ env · θ0 · conj(θ0),  θ0[aA,p,p,bB] = Σ_bond R_A R_B.    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Reshape a traced double tensor (Dl²,Dr²,Du²,Dd²) → (Dl,Dl,Dr,Dr,Du,Du,Dd,Dd).
Index pairs are (bra, ket) — bra is the faster sub-index (matches `_dl_traced`)."""
function _split_dl(E::Array{ComplexF64,4}, Dl, Dr, Du, Dd)
    return reshape(E, Dl, Dl, Dr, Dr, Du, Du, Dd, Dd)
end

"""Reshape a boundary-MPS phys leg (X²) into (bra, ket)."""
_split_mps(M::Array{ComplexF64,3}, DX) =
    reshape(M, size(M,1), DX, DX, size(M,3))   # (vL, x_bra, x_ket, vR)

# ── Charge-preserving QR of a bond site ──────────────────────────────────────

"""
    qr_reduce_h_left(A) → (Q, R)  for the LEFT site of a horizontal bond.

A has legs (p,l,r=bond,u,d).  Splits outer=(l,u,d) from inner=(p,bond):
    A[p,l,r,u,d] = Σ_a Q[l,u,d,a] · R[a,p,r]
Q is an isometry on the outer legs; R carries physical + bond.
"""
function qr_reduce_h_left(A::Array{ComplexF64,5})
    dp, Dl, Db, Du, Dd = size(A)
    Aperm = permutedims(A, (2,4,5,1,3))            # (l,u,d,p,bond)
    M = reshape(Aperm, Dl*Du*Dd, dp*Db)
    F = qr(M); Q = Matrix(F.Q); R = Matrix(F.R)
    a = size(Q, 2)
    Qt = reshape(Q, Dl, Du, Dd, a)                 # (l,u,d,a)
    Rt = reshape(R, a, dp, Db)                      # (a,p,bond)
    return Qt, Rt
end

"""
    qr_reduce_h_right(B) → (Q, R)  for the RIGHT site of a horizontal bond.

B has legs (p,l=bond,r,u,d).  Splits outer=(r,u,d) from inner=(p,bond):
    B[p,l,r,u,d] = Σ_b Q[r,u,d,b] · R[b,p,l]
"""
function qr_reduce_h_right(B::Array{ComplexF64,5})
    dp, Db, Dr, Du, Dd = size(B)
    Bperm = permutedims(B, (3,4,5,1,2))            # (r,u,d,p,bond)
    M = reshape(Bperm, Dr*Du*Dd, dp*Db)
    F = qr(M); Q = Matrix(F.Q); R = Matrix(F.R)
    b = size(Q, 2)
    Qt = reshape(Q, Dr, Du, Dd, b)                 # (r,u,d,b)
    Rt = reshape(R, b, dp, Db)                      # (b,p,bond)
    return Qt, Rt
end

# ── Left / right row environments (spectator columns) ────────────────────────

"""Absorb a spectator column j into the left row-environment.
Lenv legs (tL,bL,lket,lbra); returns new Lenv (tR,bR,rket,rbra)."""
function _spectator_left(Lenv, topj, botj, Ej, Dl,Dr,Du,Dd)
    E8  = _split_dl(Ej, Dl,Dr,Du,Dd)               # (l2,l1,r2,r1,u2,u1,d2,d1)
    tp  = _split_mps(topj, Du)                     # (tL,u2,u1,tR)
    bt  = _split_mps(botj, Dd)                     # (bL,d2,d1,bR)
    @tensor Lnew[tR,bR,rket,rbra] :=
        Lenv[tL,bL,lket,lbra] * tp[tL,u2,u1,tR] * bt[bL,d2,d1,bR] *
        E8[lbra,lket,rbra,rket,u2,u1,d2,d1]
    return Lnew
end

"""Absorb a spectator column j into the right row-environment.
Renv legs (tR,bR,rket,rbra); returns new Renv (tL,bL,lket,lbra)."""
function _spectator_right(Renv, topj, botj, Ej, Dl,Dr,Du,Dd)
    E8  = _split_dl(Ej, Dl,Dr,Du,Dd)
    tp  = _split_mps(topj, Du)
    bt  = _split_mps(botj, Dd)
    @tensor Rnew[tL,bL,lket,lbra] :=
        Renv[tR,bR,rket,rbra] * tp[tL,u2,u1,tR] * bt[bL,d2,d1,bR] *
        E8[lbra,lket,rbra,rket,u2,u1,d2,d1]
    return Rnew
end

"""
    bond_environment_h(gp, ix, iy; χ) → (env, RA, RB, QA, QB)

Reduced environment of horizontal bond (ix,iy)–(ix+1,iy):
  env[aA,bB,aA',bB'] such that
     ⟨ψ|ψ⟩ = Σ env[aA,bB,aA',bB'] θ0[aA,p,q,bB] conj(θ0[aA',p,q,bB'])
  with θ0[aA,p,q,bB] = Σ_bond RA[aA,p,bond] RB[bB,q,bond].
Uses sqrt-absorbed (symmetric-gauge) tensors.  Only the bond's two sites use
the gate-free reduced tensors RA, RB; everything else is in env.
"""
function bond_environment_h(gp::GaugePEPS, ix::Int, iy::Int; χ::Int=64)
    nxv = nx(gp)
    top = boundary_top(gp, iy; χ=χ)
    bot = boundary_bottom(gp, iy; χ=χ)
    row = _row_dl(gp, iy)                            # traced double tensors (sqrt-absorbed)

    # sqrt-absorbed site tensors of the two bond sites (gate-free)
    A = sqrt_absorbed_site(gp, ix,   iy)
    B = sqrt_absorbed_site(gp, ix+1, iy)
    QA, RA = qr_reduce_h_left(A)                     # QA(l,u,d,a)  RA(a,p,bond)
    QB, RB = qr_reduce_h_right(B)                    # QB(r,u,d,b)  RB(b,p,bond)

    DlA = size(A,2); DuA = size(A,4); DdA = size(A,5)
    DrB = size(B,3); DuB = size(B,4); DdB = size(B,5)

    # ── Left environment: spectator columns 1..ix-1 ──────────────────────
    Lenv = reshape(ComplexF64[1.0], 1,1,1,1)        # (tL,bL,lket,lbra)
    for j in 1:ix-1
        Dl,Dr,Du,Dd = isqrt.(size(row[j]))
        Lenv = _spectator_left(Lenv, top[j], bot[j], row[j], Dl,Dr,Du,Dd)
    end

    # ── Right environment: spectator columns ix+2..nx ────────────────────
    Renv = reshape(ComplexF64[1.0], 1,1,1,1)        # (tR,bR,rket,rbra)
    for j in nxv:-1:ix+2
        Dl,Dr,Du,Dd = isqrt.(size(row[j]))
        Renv = _spectator_right(Renv, top[j], bot[j], row[j], Dl,Dr,Du,Dd)
    end

    # ── Contract A column: Lenv + QA + top[ix] + bot[ix] → GA[aA,aA',tR,bR]
    tpA = _split_mps(top[ix], DuA)                  # (tL,u2,u1,tR)
    btA = _split_mps(bot[ix], DdA)                  # (bL,d2,d1,bR)
    @tensor GA[aA,aAp,tR,bR] :=
        Lenv[tL,bL,lket,lbra] * tpA[tL,u2,u1,tR] * btA[bL,d2,d1,bR] *
        QA[lket,u1,d1,aA] * conj(QA[lbra,u2,d2,aAp])

    # ── Contract B column: Renv + QB + top[ix+1] + bot[ix+1] → GB[bB,bB',tL,bL]
    tpB = _split_mps(top[ix+1], DuB)
    btB = _split_mps(bot[ix+1], DdB)
    @tensor GB[bB,bBp,tL,bL] :=
        Renv[tR,bR,rket,rbra] * tpB[tL,u2,u1,tR] * btB[bL,d2,d1,bR] *
        QB[rket,u1,d1,bB] * conj(QB[rbra,u2,d2,bBp])

    # ── Close the MPS bonds between the A and B columns ──────────────────
    @tensor env[aA,bB,aAp,bBp] := GA[aA,aAp,t,b] * GB[bB,bBp,t,b]

    return (env=env, RA=RA, RB=RB, QA=QA, QB=QB)
end

"""
    bmps_env_selftest(; …) → Bool

Validate `bond_environment_h`: the environment contracted with the gate-free
reduced two-site tensor must reproduce ⟨ψ|ψ⟩ from `peps_norm2`, for every
horizontal bond.  This is the decisive correctness check for Stage 2a.
"""
function bmps_env_selftest(; nxv::Int=3, nyv::Int=4, dg::Int=1,
                            g::Float64=1.0, t_hop::Float64=1.0, m::Float64=0.25,
                            χ::Int=64, nsteps::Int=20, noise::Float64=0.1)
    gch = zeros(Int, nxv, nyv)
    for iy in 1:nyv, ix in 1:nxv
        isodd(ix + iy) && (gch[ix, iy] = -1)
    end
    gp = init_gauge_peps(nxv, nyv, dg, gch; noise=noise)
    for _ in 1:nsteps
        trotter_step_gauge!(gp; gcoup=g, t_hop=t_hop, m=m, τ=0.05, D_max=6)
    end

    nrm_ref = real(peps_norm2(gp; χ=χ))
    println("─── boundary-MPS Stage-2a environment self-test ───")
    @printf("  reference ⟨ψ|ψ⟩ = %.8e\n", nrm_ref)
    all_ok = true
    for iy in 1:nyv, ix in 1:nxv-1
        E = bond_environment_h(gp, ix, iy; χ=χ)
        @tensor θ0[aA,p,q,bB] := E.RA[aA,p,k] * E.RB[bB,q,k]
        # env · θ0 over (aA,bB) → T[aA',bB',p,q]; then contract with conj(θ0).
        @tensor T[aAp,bBp,p,q] := E.env[aA,bB,aAp,bBp] * θ0[aA,p,q,bB]
        nrm_env = real(sum(T .* conj(permutedims(θ0, (1,4,2,3)))))
        rel = abs(nrm_env - nrm_ref) / abs(nrm_ref)
        ok = rel < 1e-6
        all_ok &= ok
        @printf("  bond h (%d,%d): env-norm = %.8e   rel.err = %.2e  %s\n",
                ix, iy, nrm_env, rel, ok ? "PASS" : "FAIL")
    end
    println(all_ok ? "  PASS: all horizontal bond environments reproduce ⟨ψ|ψ⟩" :
                     "  FAIL: environment mismatch — inspect above")
    return all_ok
end
