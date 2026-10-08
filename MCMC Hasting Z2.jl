# MCMC Hasting estimate for pc in Z2 percolation

using DataStructures
using Base.Threads
using Statistics
using Plots
using Distributions


#confere se há percolação usando Disjoint Sets
function DS_ispercolating(n::Int, p::Float64)
    # Generate boolean lattice
    lattice = rand(n, n) .< p
    
    num_sites = n * n
    v_top = num_sites + 1
    v_bottom = num_sites + 2
    
    ds = IntDisjointSets(num_sites + 2)
    
    @inbounds for r in 1:n
        row_offset = (r - 1) * n
        for c in 1:n
            if !lattice[r, c]
                continue
            end
            
            curr_id = row_offset + c
            
            # Connect top row to virtual top
            if r == 1
                union!(ds, curr_id, v_top)
            end
            
            # Connect bottom row to virtual bottom
            if r == n
                union!(ds, curr_id, v_bottom)
            end
            
            # Connect right neighbor
            if c < n && lattice[r, c + 1]
                union!(ds, curr_id, curr_id + 1)
            end
            
            # Connect down neighbor
            if r < n && lattice[r + 1, c]
                union!(ds, curr_id, curr_id + n)
            end
            
            # Check early exit condition: top and bottom connected
            if in_same_set(ds, v_top, v_bottom)
                return true
            end
        end
    end
    
    return in_same_set(ds, v_top, v_bottom)
end


#conta o número de vezes que há percolação em trials
function perc_probability(n::Int, p::Float64, trials::Int)
    count = Atomic{Int}(0)
    
    @threads for i in 1:trials
        if DS_ispercolating(n, p)
            atomic_add!(count, 1)
        end
    end
    
    return count[] / trials
end


# Tuning the proposal standard deviation based on lattice size
function adaptive_proposal_std(n::Int; sigma_ref::Float64=0.008, n_ref::Int=100)
    return sigma_ref * (n_ref / n)^(3/4)
end

#MCMC Hasting algorithm to estimate pc
function mcmc_estimate_pc(
    n::Int, 
    K::Int, 
    iterations::Int, 
    prior_dist::ContinuousUnivariateDistribution; 
    proposal_std::Float64 = adaptive_proposal_std(n)
)
    target_successes = K * 0.5  
    chain = zeros(Float64, iterations)
    
    # 1. Initialize starting value by sampling directly from the external prior
    p_current = rand(prior_dist)
    
    # Calculate initial log-likelihood and log-prior
    obs_prob_curr = perc_probability(n, p_current, K)
    k_curr = obs_prob_curr * K
    log_lik_curr = -0.5 * ((k_curr - target_successes) / (0.05 * K))^2
    log_prior_curr = logpdf(prior_dist, p_current)
    
    chain[1] = p_current
    accepted = 0
    
    println("Starting MCMC Sampling over $(iterations) iterations...")
    
    for t in 2:iterations
        # 2. Propose candidate parameter p_prop ~ Normal(p_current, proposal_std)
        p_prop = p_current + randn() * proposal_std
        
        # 3. Evaluate Log-Prior at candidate step
        log_prior_prop = logpdf(prior_dist, p_prop)
        
        # If proposed state falls outside prior support (log_prior = -Inf), reject immediately
        if isinf(log_prior_prop)
            chain[t] = p_current
            continue
        end
        
        # 4. Evaluate Log-Likelihood at candidate step
        obs_prob_prop = perc_probability(n, p_prop, K)
        k_prop = obs_prob_prop * K
        log_lik_prop = -0.5 * ((k_prop - target_successes) / (0.05 * K))^2
        
        # 5. Full Bayesian Acceptance Ratio: (Log-Likelihood + Log-Prior)
        log_alpha = (log_lik_prop + log_prior_prop) - (log_lik_curr + log_prior_curr)
        
        # 6. Accept / Reject
        if log(rand()) < log_alpha
            p_current = p_prop
            log_lik_curr = log_lik_prop
            log_prior_curr = log_prior_prop
            accepted += 1
        end
        
        chain[t] = p_current
    end
    
    acc_rate = (accepted / iterations) * 100
    println("MCMC Finished! Acceptance Rate: $(round(acc_rate, digits=2))%")
    return chain
end

# Selecting our prior for pc
my_prior = Uniform(0.40, 0.80)  #theoretical value of pc is 0.592746


# Running MCMC Hasting
n = 100            # Lattice size
K = 100            # Monte Carlo trials per proposal step
iterations = 2000 
chain = mcmc_estimate_pc(n, K, iterations, my_prior)

# marking burn-in samples
burn_in = Int(0.2 * iterations)
posterior_samples = chain[burn_in+1:end]

# descriptive results
p_c_mean   = mean(posterior_samples)
p_c_median = median(posterior_samples)
p_c_std    = std(posterior_samples)
ci_95      = quantile(posterior_samples, [0.025, 0.975])

println("\n--- POSTERIOR RESULTS (n = $n) ---")
println("Mean:     ", round(p_c_mean, digits=5))
println("Median:   ", round(p_c_median, digits=5))
println("Std Dev:  ", round(p_c_std, digits=5))
println("95% CI:   [", round(ci_95[1], digits=5), ", ", round(ci_95[2], digits=5), "]")


# trace plot
p_trace = plot(chain, xlabel="Iteration", ylabel="p", title="MCMC Trace Plot", legend=false)
vline!(p_trace, [burn_in], color=:red, linestyle=:dash)

p_hist = histogram(
    posterior_samples, 
    nbins=30, 
    normalize=:pdf, 
    xlabel="p_c", 
    ylabel="Density", 
    title="Posterior Distribution",
    color=:skyblue,
    label="Posterior"
)
vline!(p_hist, [p_c_mean], color=:red, linewidth=2, label="Mean ($(round(p_c_mean, digits=4)))")
vline!(p_hist, [0.592746], color=:black, linestyle=:dash, linewidth=2, label="Theory (0.5927)")

plot(p_trace, p_hist, layout=(2,1), size=(700, 600))
