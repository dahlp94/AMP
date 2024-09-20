using Pkg
Pkg.add("Plots")

using LinearAlgebra
using Printf
using Statistics
using Plots

function simple_LowRAMP_XX(S, Delta, RANK, max_iter=100)
    n = size(S, 1)
    x = randn(n, RANK)
    
    convergence_history = Float64[]
    iteration_count = 0
    
    for t in 1:max_iter
        B = (S * x) / sqrt(n)
        A = (x' * x) / (n * Delta)
        x_new = (A \ B')'
        
        diff = norm(x - x_new) / norm(x)
        push!(convergence_history, diff)
        
        @printf("Iteration %d: relative diff = %.6f\n", t, diff)
        
        x = x_new
        iteration_count = t
        
        if diff < 1e-6
            println("Converged!")
            break
        end
    end
    
    return x, convergence_history, iteration_count
end

# Sample data generation
n = 100
RANK = 2
Delta = 0.1

S = randn(n, n)
S = (S + S') / 2

println("Running simple_LowRAMP_XX algorithm...")
result, conv_history, iterations = simple_LowRAMP_XX(S, Delta, RANK)

# Additional statistics and analysis
println("\nAlgorithm Statistics:")
println("---------------------")
println("Number of iterations: ", iterations)
println("Final relative difference: ", conv_history[end])
println("Mean relative difference: ", mean(conv_history))
println("Median relative difference: ", median(conv_history))
println("Standard deviation of relative difference: ", std(conv_history))

# Frobenius norm of the approximation
approx_norm = norm(result * result')
original_norm = norm(S)
relative_error = norm(S - result * result') / original_norm

println("\nApproximation Quality:")
println("---------------------")
println("Frobenius norm of original matrix: ", original_norm)
println("Frobenius norm of approximation: ", approx_norm)
println("Relative Frobenius norm error: ", relative_error)

# Eigenvalue analysis
orig_eigvals = eigvals(S)
approx_eigvals = eigvals(result * result')

println("\nEigenvalue Analysis:")
println("--------------------")
println("Top 5 eigenvalues of original matrix: ", sort(orig_eigvals, rev=true)[1:5])
println("Top 5 eigenvalues of approximation: ", sort(approx_eigvals, rev=true)[1:5])

# Plotting convergence history and approximation quality
p1 = plot(1:iterations, conv_history, xlabel="Iteration", ylabel="Relative Difference", 
     title="Convergence History", yscale=:log10)

p2 = bar(["Approximation Norm", "Original Norm"], [approx_norm, original_norm], 
    title="Approximation Quality (Relative Error: $(round(relative_error, digits=4)))", 
    ylabel="Norm Value", color=[:blue :orange])

# Display the plots
display(p1)
display(p2)

# Save the plots
savefig(p1, "convergence_history.png")
savefig(p2, "approximation_quality.png")

# Optionally, create a combined plot and save it
combined_plot = plot(p1, p2, layout=(2,1), size=(800, 1000))
savefig(combined_plot, "combined_plots.png")
