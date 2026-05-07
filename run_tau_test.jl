try
    include("test_tau_scaling.jl")
catch e
    showerror(stdout, e, catch_backtrace())
    println()
    flush(stdout)
end
