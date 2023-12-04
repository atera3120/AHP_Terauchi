using Pkg

function main()
    Pkg.add([
        "JuMP",
        "HiGHS",
        "IntervalArithmetic",
        "Latexify",
        "LaTeXStrings",
        "Plots",
        "PyPlot",
        "Distributions",
        "CSV",
        "DataFrames"
        ])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
