try
    include("tensorkit_tst.jl")
catch e
    showerror(stdout, e, catch_backtrace())
end
