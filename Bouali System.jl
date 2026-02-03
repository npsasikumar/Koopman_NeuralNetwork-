using Flux
using Flux:params
using DifferentialEquations
using Plots
using LinearAlgebra
using Dates
using DataFrames
using ExcelFiles
using XLSX
using StatsBase
using Optimisers
using GlobalSensitivity, Statistics
using QuasiMonteCarlo, StableRNGs
start_time = now()

# Define the updated Rossler system
function bouali_with_control!(du, u, p, t)
	a, b, c, d, k1, k2, k3 = p
	xm, ym, zm, xs, ys, zs = u

	# Master oscillator
	du[1] = -ym -zm
	du[2] = xm + a * ym
	du[3] = b + zm * (xm - c)

	# Slave oscillator with control terms
	du[4] = -ys -zs + k1
	du[5] = xs + a * ys + k2
	du[6] = b + zs * (xs - c) + k3
end

# Initial conditions and parametes
u0 = [1.0, 0.0, 1.5, 0.5, 0.5, 0.5] # Initial states [xm, ym, zm, xs, ys, zs]

p = [3.0, 2.2, 1.0, 0.001, 0.0, 0.0, 0.0]

tspan = (0.0, 100.0)

dt = 0.001

prob = ODEProblem(bouali_with_control!, u0, tspan, p)

sol = solve(prob, Tsit5(), dt = dt, adaptive = false)

size(sol)

any(u -> any(!isfinite, u), sol.u)

# Extract master states [xm, ym, zm]
master_states = hcat([s[1:3] for s in sol.u]...)  # 3 × N matrix

function psi(x)
    xm, ym, zm = x
    return [xm, ym, zm, xm^2, ym^2, zm^2, ym*zm, xm*zm, zm*xm, 1.0]
end

X = hcat([psi(master_states[:, i]) for i in 1:(size(master_states, 2)-1)]...)

Y = hcat([psi(master_states[:, i+1]) for i in 1:(size(master_states, 2)-1)]...)

function normalize_states(X::Matrix{Float64})
    μ = mean(X; dims=2)
    σ = std(X; dims=2) .+ eps()
    return (X .- μ) ./ σ, μ, σ
end

begin
    # --- Koopman lifting ---
    function koopman_basis(x::Vector{Float64})
        vcat(x, tanh.(x), sin.(x), cos.(x), [1.0])
    end

    psi0(x::Vector{Float64}) = koopman_basis(x)

    # --- Build EDMD matrices ---
    function build_edmd_matrices(X::Matrix{Float64})
        ΨX = hcat([koopman_basis(X[:, k]) for k in 1:size(X,2)-1]...)
        ΨY = hcat([koopman_basis(X[:, k+1]) for k in 1:size(X,2)-1]...)
        ΨX, ΨY
    end

    # --- Recompute EDMD ---
    ΨX, ΨY = build_edmd_matrices(X)
    K = ΨY * pinv(ΨX)

    # --- Koopman modes for reconstruction ---
    C = X[:, 1:end-1] * pinv(ΨX)

    # --- HARD consistency check ---
    @assert length(psi0(X[:,1])) == size(K,1) == size(C,2)
end

function edmd(X::Matrix{Float64})
    @assert all(isfinite, X)

    Xn, μ, σ = normalize_states(X)
    ΨX, ΨY   = build_edmd_matrices(Xn)
    K        = koopman_operator(ΨX, ΨY)

    @assert all(isfinite, K)
    return K, μ, σ
end

function koopman_operator(ΨX::Matrix{Float64}, ΨY::Matrix{Float64}; λ::Float64 = 1e-8)
    return ΨY * ΨX' * inv(ΨX * ΨX' + λ * I)
end

x0 = X[:, 1]                # state (dimension = state_dim)

ψ0 = koopman_basis(x0)      # lifted state

println("Actual lifted next state: ", psi(master_states[:, 2]))

Ψtraj = []

"""
Compute Koopman modes for state reconstruction.

Xn   : normalized state snapshots (n × T)
ΨX   : lifted snapshots (p × T)
"""
function koopman_modes(Xn::Matrix{Float64}, ΨX::Matrix{Float64})
    return Xn[:, 1:size(ΨX, 2)] * pinv(ΨX)
end

begin
    """
    Fit an EDMD Koopman model with guaranteed correct alignment.

    Parameters
    ----------
    X : Matrix{Float64}
        State snapshot matrix (state_dim × T)

    Returns
    -------
    model : NamedTuple with fields
        K   : Koopman operator
        Φ   : Koopman modes
        μ   : state mean
        σ   : state std
        ψ   : Koopman basis function
    """
    function fit_koopman_model(X::Matrix{Float64})
        @assert all(isfinite, X)
        @assert size(X, 2) ≥ 2

        # --- normalization ---
        μ = mean(X; dims=2)
        σ = std(X; dims=2) .+ eps()
        Xn = (X .- μ) ./ σ

        # --- Koopman basis ---
        ψ(x) = vcat(
            x,
            tanh.(x),
            sin.(x),
            cos.(x),
            [1.0]
        )

        # --- build aligned data ---
        ΨX = hcat([ψ(Xn[:, k]) for k in 1:size(Xn,2)-1]...)
        ΨY = hcat([ψ(Xn[:, k+1]) for k in 1:size(Xn,2)-1]...)

        # --- Koopman operator (ridge) ---
        λ = 1e-4
        K = ΨY * ΨX' * inv(ΨX * ΨX' + λ * I)

        @assert all(isfinite, K)

        # --- Koopman modes (ALIGNED) ---
        Xn_trunc = Xn[:, 1:size(ΨX,2)]
        Φ = Xn_trunc * pinv(ΨX)

        return (
            K = K,
            Φ = Φ,
            μ = μ,
            σ = σ,
            ψ = ψ
        )
    end
end

model = fit_koopman_model(X) 

begin
	#model = fit_koopman_model(X)
	K_trunc = truncate_koopman(model.K; r = 30)
	model1 = merge(model, (; K = K_trunc))
end 

"""
Rollout Koopman model and reconstruct state trajectory.
"""
function koopman_rollout(model, x0, N)
    μ, σ = model.μ[:], model.σ[:]

    # normalize + lift
    x0n = (x0 .- μ) ./ σ
    ψk = model.ψ(x0n)

    Xrec = zeros(length(x0), N)

    for k in 1:N
        xkn = model.Φ * ψk
        Xrec[:, k] = xkn
        ψk = model.K * ψk

        if !all(isfinite, ψk)
            error("Koopman rollout unstable at step $k")
        end
    end

    return Xrec .* σ .+ μ
end

"""
Compute RMSE vs prediction horizon.
"""
function koopman_rmse_vs_horizon(model, X; max_horizon=200)
    x0 = X[:, 1]
    rmses = Float64[]

    for h in 1:max_horizon
        try
            Xrec = koopman_rollout(model, x0, h)
            push!(rmses, sqrt(mean((X[:, 1:h] .- Xrec).^2)))
        catch
            break   # stop at instability
        end
    end

    return rmses
end

ρ = maximum(abs.(eigvals(model.K)))

"""
Stabilize Koopman operator by projecting eigenvalues
onto the disk |λ| ≤ ρmax in the complex plane.
"""
function stabilize_koopman(K; ρmax=0.999)
    λ, V = eigen(K)

    λs = similar(λ)
    for i in eachindex(λ)
        r = abs(λ[i])
        if r > ρmax
            λs[i] = (ρmax / r) * λ[i]
        else
            λs[i] = λ[i]
        end
    end

    return real(V * Diagonal(λs) * inv(V))
end

ρ_before = maximum(abs.(eigvals(model.K)))

K_stable = stabilize_koopman(model.K; ρmax=0.999)

ρ_after = maximum(abs.(eigvals(K_stable)))

@show ρ_before ρ_after

λ = eigvals(model1.K)

scatter(
    real.(λ), imag.(λ),
    xlabel="Re(λ)",
    ylabel="Im(λ)",
    title="Koopman Spectrum",
    aspect_ratio=:equal,
    legend=false
)

rmse = koopman_rmse_vs_horizon(model1, X; max_horizon=300)

any(isnan, rmse), any(isinf, rmse)

any(isnan, X), any(isinf, X)

Xtest = koopman_rollout(model1, X[:,1], 50)

any(isnan, Xtest), any(isinf, Xtest)

x0n = (x0 .- model1.μ[:]) ./ model1.σ[:]

any(isnan, x0n), any(isinf, x0n)

ψ01 = model1.ψ((X[:,1] .- model1.μ[:]) ./ model1.σ[:])

x1n = model1.Φ * ψ01

any(isnan, x1n), any(isinf, x1n)

"""
Truncate Koopman operator to dominant stable modes.
"""
function truncate_koopman(K; r=40)
    U, S, V = svd(K)
    return U[:,1:r] * Diagonal(S[1:r]) * V[:,1:r]'
end

plot(
    1:length(rmse), rmse,
    xlabel="Prediction horizon (steps)",
    ylabel="RMSE",
    lw=2,
    title="Koopman Prediction Error vs Horizon"
)

K1, μ1, σ1 = edmd(master_states)

errors = [
    norm(K1 * koopman_basis(master_states[:, i]) -
         koopman_basis(master_states[:, i+1]))
    for i in 1:(size(master_states, 2)-1)
]

println("Mean Koopman prediction error: ", mean(errors))

# Train-test split
train_ratio = 0.8

train_size = Int(round(train_ratio * size(X, 2)))

test_size = size(sol)[2] - train_size

x_train = X[:, 1:train_size]

y_train = Y[:, 1:train_size]

x_test = X[:, train_size+1:end]

y_test = Y[:, train_size+1:end]

# Neural approximation of the Koopman-observable space
koopman_nn = Chain(
    Dense(10, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 10)  # outputs same dim as psi(x)
)

encoder = Chain(
    Dense(3, 32, tanh),
    Dense(32, 10)
)

K0 = Flux.params(0.01 .* randn(size(koopman_nn(x_train),1),
                             size(koopman_nn(x_train),1)))
							
loss(x, y) = Flux.Losses.mse(K01 * koopman_nn(x), koopman_nn(y))

increment_t = 0.01

opt = Flux.setup(Adam(), koopman_nn)

# Train the model
epochs = 2000

# Define loss: Koopman consistency loss
function koopman_loss(x, y)
    ψx = koopman_nn(x)
    ψy = koopman_nn(y)
    return Flux.Losses.mse(K * ψx, ψy)
end

K01 = randn(size(koopman_nn(x_train), 1),
          size(koopman_nn(x_train), 1))
		  
ps = Flux.params(koopman_nn, K01)

model01 = (; koopman_nn, K01)

opt_state = Optimisers.setup(Adam(1e-3), model01)

for epoch in 1:epochs
    grads = Flux.gradient(model) do m
        Flux.Losses.mse(
            m.K01 * m.koopman_nn(x_train),
            m.koopman_nn(y_train)
        )
    end

    opt_state, model = Optimisers.update!(opt_state, model, grads)
end

begin
	# Define unit circle
	θ = range(0, 2π, length=300)
	circle_x = cos.(θ)
	circle_y = sin.(θ)
end

begin

    eigvals_K = eigvals(K)

    p0 = Plots.scatter(
        real.(eigvals_K),
        imag.(eigvals_K),
        xlabel = "Re(λ)",
        ylabel = "Im(λ)",
        title  = "Koopman Spectrum",
        legend = false,
        aspect_ratio = 1,
        markersize = 5
    )

    Plots.plot!(
        circle_x,
        circle_y,
        lw = 2,
        lc = :red,
        label = "|λ| = 1"
    )

    savefig(p0, "fig_Koopman_spectrum.png")

    p0
end

begin
	scatter(real.(eigvals_K), imag.(eigvals_K),
        xlabel="Re(λ)", ylabel="Im(λ)",
        title="Koopman Spectrum",
        legend=false, label="Eigenvalues",
        aspect_ratio=1, markersize=5)
	# Plot eigenvalues and unit circle
	plt_koop=plot!(circle_x, circle_y, lw=2, lc=:red, label="|λ|=1")
	savefig("E:\\New folder\\Paper to Journal details\\3. IJBC\\Manuscript\\fig_Koopman_spectrum.png")
end

train_loss = loss(x_train, y_train)

test_loss = loss(x_test, y_test)

function adjust_control_with_koopman!(du, u, p, t)
    a, b, c, k1, k2, k3 = p
    xm, ym, zm, xs, ys, zs = u

    # Koopman-based prediction for master
    #ψ_master = psi([xm, ym, zm])
	ψ_master = koopman_basis([xm, ym, zm])
    #ψ_next = K * ψ_master
	ψ_next   = K * ψ_master
    #xm_next, ym_next, zm_next = ψ_next[1:3]  # first 3 are original states
	xm_next, ym_next, zm_next = ψ_next[1:3]
	
    # Compute synchronization error
    ex, ey, ez = xs - xm_next, ys - ym_next, zs - zm_next

    # PD-type control
    k1, k2, k3 = -0.1 * ex, -0.1 * ey, -0.1 * ez
    p[4:6] .= [k1, k2, k3]

    # Update dynamics
    du[1] = -ym - zm
    du[2] = xm + a * ym
    du[3] = b + zm * (xm - c)
    du[4] = -ys - zs + k1
    du[5] = xs + a * ys + k2
    du[6] = b + zs * (xs - c) + k3

	z = encoder([xm, ym, zm])
	z_next = K * z
	x̂ = decoder(z_next)

	xm_next, ym_next, zm_next = x̂
	
end

prob_koop = ODEProblem(adjust_control_with_koopman!, u0, tspan, p)


function psi(x::Vector{Float64})
    x1, x2, x3 = x

    return [
        1.0,
        x1, x2, x3,
        x1^2, x2^2, x3^2,
        x1*x2, x1*x3, x2*x3,
        # ...
        # include ALL terms used in EDMD
    ]
end

length(psi([1.0, 1.0, 1.0])) == size(K, 1)

sol_koop = solve(prob_koop, Tsit5(), dt = 0.001, adaptive = false)

master_koop = [s[1:3] for s in sol_koop.u]

slave_koop = [s[4:6] for s in sol_koop.u]

sync_error_koop = [norm(master_koop[i] - slave_koop[i]) for i in 1:length(master_koop)]

plot(sol_koop.t, hcat(master_koop...)', xlabel = "Time (s)", ylabel="System State", label=["xm" "ym" "zm"], title="Master System")

plot(sol_koop.t, hcat(slave_koop...)', xlabel = "Time (s)", ylabel="System State", label=["xs" "ys" "zs"], title="Slave System", linestyle=:dash)

master_mat = reduce(vcat, master_koop')

slave_mat = reduce(vcat, slave_koop')

minval = min(minimum(master_mat), minimum(slave_mat))

maxval = max(maximum(master_mat), maximum(slave_mat))

plot(heatmap(master_mat, title="Master", clim=(minval, maxval), colorbar=false), heatmap(slave_mat, title="Slave", clim=(minval, maxval)),)

diff_mat = abs.(master_mat - slave_mat)

heatmap(diff_mat, title="|Master - Slave|", xlabel="Variables", ylabel="Time", colorbar_title="|Δ|")

plot(sync_error_koop, yscale=:log10, title="Koopman–Neural Synchronization Error", xlabel = "Time Step", ylabel = "‖Master - Slave‖", label = "Error(t)", legend = :topright)

println("Mean synchronization error: ", mean(sync_error_koop))

println("Final error: ", sync_error_koop[end])

K_param = params(randn(10,10))

loss_joint(x, y) = Flux.Losses.mse(K_param * koopman_nn(x), koopman_nn(y))

opt_joint = Flux.setup(Adam(1e-3), Flux.params(koopman_nn, K_param))

# Synchronization error vs time (log scale to show convergence)
plot(sol_koop.t, sync_error_koop,
     yscale = :log10,
     linewidth = 2,
     linecolor = :blue,
     label = "‖e(t)‖ = ‖x_slave - x_master‖",
     xlabel = "Time",
     ylabel = "Synchronization Error (log scale)",
     title = "Koopman–Neural Synchronization Convergence",
     legend = :topright,
     grid = true)
	 
# Annotate the final value
annotate!([(maximum(sol_koop.t)*0.7, minimum(sync_error_koop)*1.5,
           text("Error → $(round(sync_error_koop[end], sigdigits=3))", :black, 10))])
		   
# Save high-resolution figure for paper
savefig("E:\\New folder\\Paper to Journal details\\3. IJBC\\Manuscript\\figKoopman_Neural_Synchronization_Error2.png")

begin
	# Extract individual errors
	ex = [abs(m[1] - s[1]) for (m, s) in zip(master_koop, slave_koop)]
	ey = [abs(m[2] - s[2]) for (m, s) in zip(master_koop, slave_koop)]
	ez = [abs(m[3] - s[3]) for (m, s) in zip(master_koop, slave_koop)]
end

begin
	# Plot all on log scale
	plot(sol_koop.t, ex, yscale=:log10, lw=2, lc=:red, label="|ex|")
	plot!(sol_koop.t, ey, lw=2, lc=:green, label="|ey|")
	plot!(sol_koop.t, ez, lw=2, lc=:blue, label="|ez|",
      xlabel="Time", ylabel="State Error (log scale)",
      title="Component Synchronization Errors",
      legend=:topright, grid=true)
end

savefig("E:\\New folder\\Paper to Journal details\\3. IJBC\\Manuscript\\Component_Synchronization_Errors.png")

eigvals_K1 = eigvals(K)

abs_vals = abs.(eigvals_K1)

θ1 = range(0, 2π, length=300)

circle_x1, circle_y1 = cos.(θ1), sin.(θ1)

begin
	scatter(real.(eigvals_K1), imag.(eigvals_K1),
        marker_z = abs_vals, c = :viridis, ms = 8,
        xlabel="Re(λ)", ylabel="Im(λ)",
        title="Koopman Spectrum (Colored by |λ|)",
        aspect_ratio=1, legend=false)
	plot!(circle_x1, circle_y1, lc=:red, lw=2, label="|λ|=1")
	savefig(p4, "fig_Koopman_spectrum.png")
end

# --- Step 1: Define simulation function for Koopman-Neural synchronization ---
function simulate_Koopman_sync(p)
    try
        a, b, c, k_gain = p
        local_params = [a, b, c, 0.0, 0.0, 0.0]
        prob_local = ODEProblem(adjust_control_with_koopman!, u0, (0.0, 100.0), local_params)
        sol_local = solve(prob_local, Tsit5(), dt=0.01, adaptive=false)
        
        master = [s[1:3] for s in sol_local.u]
        slave  = [s[4:6] for s in sol_local.u]
        sync_error = [norm(m - s) for (m, s) in zip(master, slave)]
        mean_error = mean(sync_error)
        
        # Return a finite scalar
        return isfinite(mean_error) ? mean_error : 1e6  # penalize divergence
    catch
        return 1e6  # return large value for failed runs
    end
end

# Automatically use state-space min/max as parameter bounds
lb = [minimum(master_mat), minimum(master_mat), minimum(master_mat), 0.01]

ub = [maximum(master_mat), maximum(master_mat), maximum(master_mat), 1.0]

begin 
	# Sobol sampling and GSA
	n = 2000  # use 100–2000 depending on compute power
	sampler = SobolSample()
	A, B = QuasiMonteCarlo.generate_design_matrices(n, lb, ub, sampler)

	# Run global sensitivity
	function simulate_Koopman_sync_batch(X)
    	# X is a matrix of parameter samples (each column = one sample)
    	results = zeros(size(X, 2))
    	for i in 1:size(X, 2)
        	p = X[:, i]
        	results[i] = simulate_Koopman_sync(p)  # Each run gives a scalar
    	end
    	return results
	end
end

# Now run global sensitivity correctly:
sobol_result = gsa(simulate_Koopman_sync_batch, Sobol(), A, B)

println("First-order indices (S1): ", sobol_result.S1)

println("Total-order indices (ST): ", sobol_result.ST)

# Plot the indices
p3 = bar(sobol_result.S1, title = "First-order Sobol Sensitivity (Koopman–Neural)",
    xlabel = "Parameters [a, b, c, k_gain]",
    ylabel = "Sensitivity Index",
    legend = false)
	
savefig(p3, "fig_Sobol_S11.png")

# Create and assign the plot to a variable
p1 = bar(sobol_result.S1,
    title = "First-order Sobol Sensitivity (Koopman–Neural)",
    xlabel = "Parameters [a, b, c, k_gain]",
    ylabel = "Sensitivity Index",
    legend = false,
    color = :dodgerblue)

# Save the figure to file (choose format: .png, .pdf, .svg, .eps, etc.)
savefig(p1, "fig_Sobol_S1.png")

p2 = bar(sobol_result.ST,
    title = "Total Sobol Sensitivity (Koopman–Neural)",
    xlabel = "Parameters [a, b, c, k_gain]",
    ylabel = "Total Sensitivity Index",
    legend = false,
    color = :orange)
	
savefig(p2, "fig_Sobol_ST.png")

param_names = ["a", "b", "c", "k_gain"]

p4 = bar(sobol_result.S1[:], xticks=(1:4, param_names),
    title="First-order Sobol Sensitivity (Koopman–Neural)",
    xlabel="Parameters", ylabel="Sensitivity Index", legend=false)
	
savefig(p4, "fig_Sobol_last.png")
