outfile = joinpath(@__DIR__, "test_results.txt")
open(outfile, "w") do io
    println(io, "=== Starting test ===")
    flush(io)
    try
        include(joinpath(@__DIR__, "tensorkit_tst.jl"))
        println(io, "File loaded OK")
        flush(io)
        
        H_plaq_4 = build_plaquette_4site(g2=1.0)
        println(io, "H_plaq_4 size = $(size(H_plaq_4))")
        flush(io)
        
        peps = init_checkerboard(V_phys, V_bond; nf_A=0, nbr_A=0, nbu_A=0, nf_B=1, nbr_B=0, nbu_B=0)
        println(io, "PEPS initialized")
        flush(io)
        
        G_plaq = exp(-0.01im .* H_plaq_4)
        println(io, "Gate computed")
        flush(io)
        
        update_plaquette!(peps, G_plaq, D_max)
        println(io, "update_plaquette! SUCCESS")
        println(io, "A size = $(size(convert(Array, peps.A)))")
        println(io, "B size = $(size(convert(Array, peps.B)))")
        println(io, "λh = $(peps.λh)")
        println(io, "λv = $(peps.λv)")
        
        gl = measure_gauss_law(peps)
        println(io, "After plaquette: ΔG² = $(gl.var_G)")
    catch e
        println(io, "ERROR: $e")
        showerror(io, e, catch_backtrace())
        println(io)
    end
    println(io, "=== Done ===")
end
