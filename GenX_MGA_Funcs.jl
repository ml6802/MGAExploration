using JuMP, CSV, DataFrames, Random, Combinatorics

function hsj_obj(new_point::AbstractArray, indic::AbstractArray)
    nrow, ncol = size(new_point)
    for j in 1:ncol, k in 1:nrow
		if new_point[k, j] >= 0.01
			indic[k,j] += 1
		end
    end
    return indic
end

function print_dists(dists::AbstractVector, outpath::AbstractString)
    file = joinpath(outpath,"MGAdists.csv")
    its = collect(1:length(dists))
    distdf = DataFrame(Iteration = its, Avg_Dist = dists)
    CSV.write(file,distdf)
end

function est_chull_vol(points::AbstractArray)
	(ntt, nz, nit) = size(points)
	clustered = fill(0.0, (ntt, nit))
	clustered = sum(points[:,i,:] for i in 1:nz)
	println(clustered)

	pairs = collect(combinations(1:ntt, 2))
	pairwise_caps = fill(0.0, (nit, 2))
	areas = Vector{Float64}(undef, 0)
	tot_area = 0.0
	for i in eachindex(pairs)
		pairwise_caps[:,1] = view(clustered,pairs[i][1],:)'
		pairwise_caps[:,2] = view(clustered,pairs[i][2],:)'
		uni_pc = uniques(pairwise_caps)
		poly = polyhedron(vrep(Matrix(uni_pc)))
		vol = 0.0
		vol = Polyhedra.volume(poly)
        push!(areas, vol)
	end
	tot_area = sum(areas[i] for i in eachindex(areas))
	println("Volume is: "*string(tot_area))
	return tot_area
end
function uniques(points::AbstractArray)
    pointst = transpose(points)
    nrow, ncol = size(pointst)
    placeholder = fill(-1.0,ncol)
    count = 1
    while count < length(placeholder)
        """
        for i in 1:nrow
            if isnan(pointst[i,j])
                pointst[i,j] = 0.0
            end
        end
        """
        if sum(pointst[:,count]) == 0.0
            pointst = pointst[:,1:end .!= count]
            ncol = ncol-1
            pop!(placeholder)
        end
        count += 1
    end
    uniques = fill(-1.0, (nrow, ncol))
    counter=0
    for i in 1:ncol
        for k in 1:ncol
            if isapprox(pointst[:,i],uniques[:,k],atol=0.1)
                break
            elseif k == ncol
                counter = counter + 1
                uniques[:,counter] = pointst[:,i]
            end
        end
    end
    uniques = uniques[1:end, 1:counter]
    uniquesT = transpose(uniques)
    println("Done with uniques")
    return uniquesT
end

function hsj_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)

    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 1
        # Start MGA Algorithm
	    println("MGA Module")
		println("HSJ Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		# Setup storage
		Indic = fill(0, (length(TechTypes), Z))
		point_list = fill(0.0, (length(TechTypes), Z, setup["ModelingToGenerateAlternativeIterations"]+1))
		point_list[:,:,1] = value.(EP[:vSumvP])
		vols = fill(0.0, setup["ModelingToGenerateAlternativeIterations"])
		"""
	    ### Variables ###

	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type

		
        # Constraint to compute total generation in each zone from a given Technology Type
	    @constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
	    for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
	    ### End Variables ###
		"""

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )


	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_hsj = joinpath(path, "MGAResults_hsj")
	    if !(isdir(outpath_hsj))
	    	mkdir(outpath_hsj)
	    end



	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()

	    print("Starting the first MGA iteration")

	    for i in 1:setup["ModelingToGenerateAlternativeIterations"]

	    	# Create hsj coefficients for the generators that we want to include in the MGA run for the given budget
	    	Indic = hsj_obj(point_list[:,:,i],Indic) #rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])))

	    	### Maximization objective
	    	@objective(EP, Min, sum(Indic[tt,z] * EP[:vSumvP][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP)

            # Create path for saving MGA iterations
	    	mgaoutpath_hsj = joinpath(outpath_hsj, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP, mgaoutpath_hsj, setup, inputs)
			point_list[:,:,i+1] = value.(EP[:vSumvP])
			est_vol = est_chull_vol(point_list[:,:,1:i+1])
			vols[i] = est_vol
	    end

	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_hsj)
	    ### End MGA Iterations ###
	end
	
end

function heuristic_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 0
        # Start MGA Algorithm
	    println("MGA Module")
		println("Heuristic Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
        outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")
	    
	    # Create random coefficients for the generators that we want to include in the MGA run for the given budget
	    pRand = rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])),setup["ModelingToGenerateAlternativeIterations"])

	    Threads.@threads for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)

	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i])
			push!(vols, est_vol)

	    	### Minimization objective
	    	@objective(EP_c, Min, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			push!(vols, est_vol)
	    end
	    """
	    
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
	    @expression(EP, eTotEms, sum(EP[:eEmissionsByZone][i,t] for i in 1:Z, t in 1:T))
    	@objective(EP, Min, eTotEms)

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+2))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,2*i+2] = value.(EP[:vSumvCap])
		est_vol = est_chull_vol(point_list[:,:,1:2*i+2])
		push!(vols, est_vol)
	    
	    
	    println(point_list)
        est_vol = est_chull_vol(point_list)
		push!(vols, est_vol)
		"""
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end

function heuristic_combo(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 7
        # Start MGA Algorithm
	    println("MGA Module")
		println("Heuristic/CapMM Combo Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0
		counter = 0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
        outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")
	    
	    # Create random coefficients for the generators that we want to include in the MGA run for the given budget
	    pRand = rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])),Int64(ceil(setup["ModelingToGenerateAlternativeIterations"]/2)))
	    pBrack = unique_int(rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),setup["ModelingToGenerateAlternativeIterations"]))
        check_it_a_ag!(pBrack,Int64(ceil(setup["ModelingToGenerateAlternativeIterations"]/2)))
        

	    Threads.@threads for i in 1:Int64(ceil(setup["ModelingToGenerateAlternativeIterations"]/2))
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)

	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i])
			push!(vols, est_vol)

	    	### Minimization objective
	    	@objective(EP_c, Min, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			push!(vols, est_vol)
			counter += 1
	    end
	    
        Threads.@threads for i in counter:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)
            
	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pBrack[tt,i-counter+1] * sum(EP_c[:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i])
			push!(vols, est_vol)
			
			### Minimization objective
	    	@objective(EP_c, Min, sum(pBrack[tt,i-counter+1] * sum(EP_c[:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			push!(vols, est_vol)
	    end

		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end


function check_it_a_disag!(a::AbstractArray, iterations::Int64)
    (r,c,i) = size(a)
    if iterations < i
        a = a[1:r,1:c,1:iterations]
        return a
    else
        println("Error")
    end
end

function check_it_a_ag!(a::AbstractArray, iterations::Int64)
    (r,i) = size(a)
    if iterations < i
        a = a[1:r,1:iterations]
        return a
    else
        println("Error")
    end
end

function unique_int(points::AbstractArray)
    pointst = transpose(points)
    nrow, ncol = size(pointst)

    uniques = fill(-2, (nrow, ncol))
    counter=0
    for i in 1:ncol
        for k in 1:ncol
            if pointst[:,i]==uniques[:,k]
                break
            elseif k == ncol
                counter = counter + 1
                uniques[:,counter] = pointst[:,i]
            end
        end
    end
    uniques = uniques[1:end, 1:counter]
    uniquesT = transpose(uniques)
    println("Done with uniques")
    return uniquesT
end

function Disag_capminmax_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 5
        # Start MGA Algorithm
	    println("MGA Module")
		println("Spatially Disaggregated Capacity Min/Max Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
	    outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end


	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")
    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
    	pRand = rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])), setup["ModelingToGenerateAlternativeIterations"])

	    Threads.@threads for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)
            
	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i])
			push!(vols, est_vol)
			
			### Minimization objective
	    	@objective(EP_c, Min, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			push!(vols, est_vol)
	    end
	    """
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
    	@objective(EP, Min, sum(sum(EP[:eEmissionsByZone][i,t] for i in 1:Z) for t in 1:T))

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+1))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,i+1] = value.(EP[:vSumvCap])
		est_vol = est_chull_vol(point_list[:,:,1:2i+2])
		push!(vols, est_vol)
	    """
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end


function Ag_capminmax_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 6
        # Start MGA Algorithm
	    println("MGA Module")
		println("Tech Aggregated Capacity Min/Max Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)
	    println(Least_System_Cost)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		est_vol = 0.0
		vols = Vector{Float64}(undef, 0)

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )
	    println(Least_System_Cost*(1+slack))

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
	    outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end


	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
        pRand = unique_int(rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),setup["ModelingToGenerateAlternativeIterations"]))
        check_it_a_ag!(pRand,setup["ModelingToGenerateAlternativeIterations"])
        	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")

	    Threads.@threads for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)
            
	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,i] * sum(EP_c[:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i])
			push!(vols, est_vol)
			
			### Minimization objective
	    	@objective(EP_c, Min, sum(pRand[tt,i] * sum(EP_c[:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			push!(vols, est_vol)
	    end
	    """
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
    	@objective(EP, Min, sum(sum(EP[:eEmissionsByZone][i,t] for i in 1:Z) for t in 1:T))

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+1))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,2*i+2] = value.(EP[:vSumvCap])
		est_vol = est_chull_vol(point_list[:,:,1:2*i+2])
		push!(vols, est_vol)
	    """
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###

	end
end

function Disag_capmax_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 3
        # Start MGA Algorithm
	    println("MGA Module")
		println("Spatially Disaggregated Capacity Max Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end


	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")
    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
    	pRand = rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])), setup["ModelingToGenerateAlternativeIterations"])

	    Threads.@threads for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)
            
	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,z,i] * EP_c[:vSumvCap][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:i])
			push!(vols, est_vol)
	    end
	    """
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
    	@objective(EP, Min, sum(sum(EP[:eEmissionsByZone][i,t] for i in 1:Z) for t in 1:T))

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+1))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,i+1] = value.(EP[:vSumvCap])
		est_vol = est_chull_vol(point_list[:,:,1:i+1])
		push!(vols, est_vol)
		"""
	    
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end


function Ag_capmax_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)
    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 4
        # Start MGA Algorithm
	    println("MGA Module")
		println("Tech Aggregated Capacity Max Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, setup["ModelingToGenerateAlternativeIterations"]+2))
		point_list[:,:,1] = value.(EP[:vSumvCap])
		vols = Vector{Float64}(undef, 0)
		est_vol = 0.0

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end


	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
        pRand = unique_int(rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),2*setup["ModelingToGenerateAlternativeIterations"]))
        check_it_a_ag!(pRand,setup["ModelingToGenerateAlternativeIterations"])
        	    
	    # workers
	    # np_ids
	    

        # create total job list
        # 

	    print("Starting the first MGA iteration")

	    Threads.@threads for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	        EP_c = copy(EP)
	        set_optimizer(EP_c, CPLEX.Optimizer)
            
	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,i] * sum(EP_c[:vSumvCap][tt,z] for z in 1:Z) for tt in 1:length(TechTypes)))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,i] = value.(EP_c[:vSumvCap])
			est_vol = est_chull_vol(point_list[:,:,1:i])
			push!(vols, est_vol)
	    end
	    """
	    # Beginning of Bracketing Runs - might add in separate function later
	    ### Minimization objective
    	@objective(EP, Min, sum(sum(EP[:eEmissionsByZone][i,t] for i in 1:Z) for t in 1:T))

    	# Solve Model Iteration
    	status = optimize!(EP)

        # Create path for saving MGA iterations
    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i+1))

    	# Write results
    	write_outputs(EP, mgaoutpath_min, setup, inputs)
		point_list[:,:,i+1] = value.(EP[:vSumvCap])
	
		est_vol = est_chull_vol(point_list[:,:,1:i+1])
		push!(vols, est_vol)
		"""
	    
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
end

function sequential_heuristic_mga(EP::Model, path::AbstractString, setup::Dict, inputs::Dict, outpath::AbstractString)

    if setup["ModelingToGenerateAlternatives"]==1 && setup["MGAMethod"] == 2
        # Start MGA Algorithm
	    println("MGA Module")
		println("Heuristic Method")

	    # Objective function value of the least cost problem
	    Least_System_Cost = objective_value(EP)

	    # Read sets
	    dfGen = inputs["dfGen"]
	    T = inputs["T"]     # Number of time steps (hours)
	    Z = inputs["Z"]     # Number of zonests
	    G = inputs["G"]

	    # Create a set of unique technology types
	    TechTypes = unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])
	    println("TT is: $TechTypes")
	    println("Z is: $Z")

	    # Read slack parameter representing desired increase in budget from the least cost solution
	    slack = setup["ModelingtoGenerateAlternativeSlack"]

		
		point_list = fill(0.0, (length(TechTypes), Z, 2*setup["ModelingToGenerateAlternativeIterations"]+1))
		point_list[:,:,1] = value.(EP[:vSumvP])
		vols = Vector{Float64}(undef, 0)

	    ### Variables ###
		"""
	    @variable(EP, vSumvP[TechTypes = 1:length(TechTypes), z = 1:Z] >= 0) # Variable denoting total generation from eligible technology of a given type
		# Constraint to compute total generation in each zone from a given Technology Type
		@constraint(EP,cGeneration[tt = 1:length(TechTypes), z = 1:Z], vSumvP[tt,z] == sum(EP[:vP][y,t] * inputs["omega"][t]
		for y in dfGen[(dfGen[!,:Resource_Type] .== TechTypes[tt]) .& (dfGen[!,:Zone] .== z),:][!,:R_ID], t in 1:T))
		"""
	    ### End Variables ###

	    ### Constraints ###

	    # Constraint to set budget for MGA iterations
	    @constraint(EP, budget, EP[:eObj] <= Least_System_Cost * (1 + slack) )

	    ### End Constraints ###

	    ### Create Results Directory for MGA iterations
        outpath_max = joinpath(path, "MGAResults_max")
	    if !(isdir(outpath_max))
	    	mkdir(outpath_max)
	    end
        outpath_min = joinpath(path, "MGAResults_min")
	    if !(isdir(outpath_min))
	    	mkdir(outpath_min)
	    end

	    ### Begin MGA iterations for maximization and minimization objective ###
	    mga_start_time = time()
        EP_c = copy(EP) # Take out
	    set_optimizer(EP_c, CPLEX.Optimizer)
	    print("Starting the first MGA iteration")

	    for i in 1:setup["ModelingToGenerateAlternativeIterations"]
	    	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
	    	pRand = rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])))

	    	### Maximization objective
	    	@objective(EP_c, Max, sum(pRand[tt,z] * EP_c[:vSumvP][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_max = joinpath(outpath_max, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_max, setup, inputs)
			point_list[:,:,2*i] = value.(EP_c[:vSumvP])
			est_vol = est_chull_vol(point_list[:,:,1:2*i])
			push!(vols, est_vol)

	    	### Minimization objective
	    	@objective(EP_c, Min, sum(pRand[tt,z] * EP_c[:vSumvP][tt,z] for tt in 1:length(TechTypes), z in 1:Z ))

	    	# Solve Model Iteration
	    	status = optimize!(EP_c)

            # Create path for saving MGA iterations
	    	mgaoutpath_min = joinpath(outpath_min, string("MGA", "_", slack,"_", i))

	    	# Write results
	    	write_outputs(EP_c, mgaoutpath_min, setup, inputs)
			point_list[:,:,2*i+1] = value.(EP_c[:vSumvP])
			est_vol = est_chull_vol(point_list[:,:,1:2*i+1])
			push!(vols, est_vol)
	    end
	    println(point_list)
        est_vol = est_chull_vol(point_list)
		push!(vols, est_vol)
		println("Final Volume Estimate is: $est_vol")
	    total_time = time() - mga_start_time
	    print_dists(vols, outpath_max)
	    ### End MGA Iterations ###
	end
	
end


function combo_vectors(inputs::Dict, setup::Dict)
	# Read sets
	dfGen = inputs["dfGen"]

	# Create random coefficients for the generators that we want to include in the MGA run for the given budget
	pRand = rand(length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),length(unique(dfGen[!,:Zone])),Int64(ceil(setup["ModelingToGenerateAlternativeIterations"]/4)))
	pBracket = unique_int(rand(-1:1,length(unique(dfGen[dfGen[!, :MGA] .== 1, :Resource_Type])),setup["ModelingToGenerateAlternativeIterations"]))
	check_it_a_ag!(pBracket,Int64(ceil(3*setup["ModelingToGenerateAlternativeIterations"]/4)))
	return pRand, pBracket
end


function make_rand_vecs(iterations::Int64, TechTypes::Int64, Zones::Int64)
    vecs = rand(Float64,(TechTypes,Zones,iterations))
    return vecs
end

function make_capMM_vecs(iterations::Int64, TechTypes::Int64, Zones::Int64)
    vecs = unique_int(rand(-1:1,TechTypes,2*iterations))
    check_it_a_ag!(vecs,iterations)
    cap_vecs = convert_ag_to_disag(vecs,Zones)
    return cap_vecs
end

function make_combo_vecs(iterations::Int64, TechTypes::Int64, Zones::Int64, ratio::Float64)
    rand_vecs = make_rand_vecs(ceil(Int64,iterations*ratio),TechTypes,Zones)
    cap_vecs = make_capMM_vecs(floor(Int64,iterations*(1-ratio)),TechTypes, Zones)
	println(cap_vecs)
    vecs = cat(rand_vecs,cap_vecs,dims=3)
    vecs = vecs[:,:,1:iterations]
    return vecs
end

function convert_ag_to_disag(ag_vecs::AbstractArray, Zones::Int64)
    (techs,iterations) = size(ag_vecs)
    vecs = Array{Float64,3}(undef,(techs,Zones,iterations))
    for i in 1:iterations
        for j in 1:techs
			vecs[j,:,i] .= ag_vecs[j,i]
        end
    end
    return vecs
end

function test_combo()
	it=5
	tt = 3
	z = 4
	r = 0.25
	vecs=make_combo_vecs(it,tt,z,r)
	println(size(vecs))
end

function check_it_a_ag!(a::AbstractArray, iterations::Int64)
    (r,i) = size(a)
    if iterations < i
        a = a[1:r,1:iterations]
        return a
    else
        println("Error")
    end
end
