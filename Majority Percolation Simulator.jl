using Random
using DataStructures
using Plots

function PercZ2(lattice::BitMatrix; title_prefix="Lattice")
    # Disjoint Set Initialization
    n = size(lattice, 1)
    num_sites = n * n
    v_top = num_sites + 1
    v_bottom = num_sites + 2
    ds = IntDisjointSets(num_sites + 2)

    # Disjoint Set
    @inbounds for r in 1:n
        row_offset = (r - 1) * n
        for c in 1:n
            if !lattice[r, c]
                continue
            end

            curr_id = row_offset + c

            if r == 1
                union!(ds, curr_id, v_top)
            end
            if r == n
                union!(ds, curr_id, v_bottom)
            end

            if c < n && lattice[r, c + 1]
                union!(ds, curr_id, curr_id + 1)
            end

            if r < n && lattice[r + 1, c]
                union!(ds, curr_id, curr_id + n)
            end
        end
    end

    # Percolation Check
    has_percolated = in_same_set(ds, v_top, v_bottom)

    # Display Matrix
    display_mat = zeros(Int, n, n)

    if has_percolated   # Highlight spanning cluster
        for r in 1:n
            row_offset = (r - 1) * n
            for c in 1:n
                if lattice[r, c]
                    curr_id = row_offset + c
                    if in_same_set(ds, curr_id, v_top)
                        display_mat[r, c] = 2
                    else
                        display_mat[r, c] = 1
                    end
                end
            end
        end
        cmap = cgrad([:black, :white, :yellow], [0.0, 0.5, 1.0], categorical=true)
    
    else  # Color clusters cleanly by root ID
        root_map = Dict{Int, Int}()
        cluster_count = 0

        for r in 1:n
            row_offset = (r - 1) * n
            for c in 1:n
                if lattice[r, c]
                    curr_id = row_offset + c
                    root = find_root!(ds, curr_id)
                    
                    # Ignore virtual nodes if they captured finite edge sites
                    if root != v_top && root != v_bottom
                        if !haskey(root_map, root)
                            cluster_count += 1
                            root_map[root] = cluster_count
                        end
                        display_mat[r, c] = root_map[root]
                    else
                        # Default fallback for un-percolated edge clusters connected to virtual nodes
                        display_mat[r, c] = 1
                    end
                end
            end
        end
        cmap = :turbo
    end

    # Plotting
    plt = heatmap(
        display_mat,
        c = cmap,
        colorbar = false,
        ticks = false,
        framestyle = :box,
        showaxis = false,
        title = title_prefix,
        aspect_ratio = 1
    )

    return plt
end

function MajorPerc(n::Int, p::Float64, l::Float64, T::Float64)
    # Initial state
    lattice = rand(n, n) .< p
    p0 = PercZ2(lattice, "Initial")

    # Point Update Queue
    pq = PriorityQueue{Tuple{Int,Int}, Float64}()

    for i in 1:n, j in 1:n
        first_time = -log(rand()) / l       # Exponential waiting time
        enqueue!(pq, (i, j), first_time)
    end

    # Neighbors
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while !isempty(pq)
        ((i, j), current_time) = dequeue_pair!(pq)  # Pull from queue

        if current_time > T                         # Stop at time T
            break
        end

        active_neighbors = 0                        # Find Majority
        for (di, dj) in neighbors
            ni = mod1(i + di, n)
            nj = mod1(j + dj, n)
            active_neighbors += lattice[ni, nj]
        end

        if active_neighbors > 2                     # Majority dynamics rule
            lattice[i, j] = true
        elseif active_neighbors < 2
            lattice[i, j] = false
        end

        next_time = current_time - log(rand()) / l    # Recalculate next update
        enqueue!(pq, (i, j), next_time)
    end

    p1 = PercZ2(lattice, "Final")

    plot(p0, p1, layout=(1, 2), size=(800, 400))
end

MajorPerc(2000, 0.56, 1.0, 200.0)