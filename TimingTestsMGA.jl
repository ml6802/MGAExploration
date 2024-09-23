using BenchmarkTools, JuMP, CPLEX, DataFrames, Combinatorics, Random, QHull, Polyhedra, CDDLib, LinearAlgebra, CSV, DelimitedFiles, AngleBetweenVectors, Statistics, Base.Threads, Plots
include("MGASandbox.jl")
using .MGASandbox

function LP_3D()
    model = Model(CPLEX.Optimizer)
    @variable(model, x[1:3] >= 0)
    @constraints(model, begin
        c1, x[1] + x[2] + x[3] >= 2
        c2, x[1] <= 3
        c3, 2*x[2] + 3*x[3] <= 5
    end)
    @objective(model, Min, model[:x][1] + 2*model[:x][2] + 2*model[:x][3])
    optimize!(model)
    println("Model created")
    nvars = all_variables(model)
    return model, nvars
end


""" Random N dimensional LP generator """

function LP_N(nvars::Int64)
    nconst = 2*nvars # more constraints than variables
    model = Model(CPLEX.Optimizer)
    A = rand(1:9,(nconst, nvars))
    b = rand(1:9,nconst)
    c = rand(1:9,nvars)
    @variable(model, x[1:nvars] >= 0)
    @constraint(model, con, A * x .>= b)
    #@constraint(model, con2, x .<= 10) # constrain all variables to be of reasonable size
    @objective(model, Min, sum(c[i]*x[i] for i in 1:nvars))
    optimize!(model)
    vars = all_variables(model)
    return model, vars
end

## Add uncertain LP here - perturb by certain percentage a few constraint parameters/obj coefficients and rerun


""" Timed MGA Functions """

function timed_hsj(Orig::MGASandbox.MGAInfo, iterations::Int64, slack::Float64)
    ender = 0
    BA = false
    N = false
    solve_time = 0.0
    obj_time = 0.0
    method = 1
    dim = length(Orig.nvars)
    model_copy = copy(Orig.orig_model)
    set_silent(model_copy)
    set_optimizer(model_copy, CPLEX.Optimizer)
    model_copy = MGASandbox.set_slack(model_copy, Orig.z, slack)

    start_time = time_ns()
    New_Point_List, New_z_list, new_p_list = MGASandbox.initialize_lists(Orig.point_list, Orig.z, iterations, Orig.nvars)
    current_point = fill(-1.0, length(Orig.nvars))
    indic = fill(0, length(Orig.nvars))
    indic_list = fill(0, iterations, length(Orig.nvars))
    init_time = time_ns() - start_time
    counter = 0
    avg_dist = 0
    conv_vec= Vector{Float64}(undef,0)
    prev_avg_dist = 0
    conv = 1
    nuni = 0
    conv_crit = 0.1
    dists = fill(-1.0, iterations)
    
    for i in 1:iterations
        prev_avg_dist = avg_dist
        counter += 1
        loop_time = time_ns()
        indic = MGASandbox.hsj_obj(New_Point_List, Orig.nvars, indic)
        indic_list[i,:] = value.(-1*indic)
        println(value.(-1*indic))
        @objective(model_copy, Min, sum(indic[j]*all_variables(model_copy)[j] for j in eachindex(indic)))
        obj_time = time_ns() - loop_time
        init_time2 = time_ns()
        optimize!(model_copy)
        println(value.(model_copy[:x]))
        solve_time = time_ns() - init_time2
        New_Point_List, New_z_list, new_p_list, current_point = MGASandbox.update_iteration(New_Point_List, New_z_list, new_p_list, i, model_copy, BA, N)
        avg_dist = est_chull_vol(New_Point_List) #evaluate_manhattan_dist(current_point,New_Point_List)
        dists[i] = avg_dist
        """
        if mod(i,10) == 0
            current_10_dists = rolling_avg_dist(dists,i)
            conv = abs(current_10_dists-prev_10_dists)
            println("Conv is "*string(conv))
            push!(conv_vec,conv)
            if current_10_dists <= conv_crit
                break
            end
        end
        
        prev_10_dists = current_10_dists
        """
    end
    tot_time = time_ns() - start_time
    times = MGASandbox.timing(init_time, obj_time, solve_time, tot_time, 0, counter)
    print_dists(dists,method, dim, conv_vec)
    New_Point_List, New_z_list, new_p_list,nuni = find_unique(New_Point_List, New_z_list, new_p_list)
    return New_z_list, New_Point_List, new_p_list, times 
end

"""
function timed_distance(Orig::MGASandbox.MGAInfo, iterations::Int64, slack::Float64) ## Solver and norm issues - manually reformulate as MILP and change to cplex
    ender = 0
    BA = false
    N = false
    method = 2
    solve_time = 0.0
    obj_time = 0.0
    model_copy = copy(Orig.orig_model)
    set_silent(model_copy)
    set_optimizer(model_copy, CPLEX.Optimizer)
    model_copy = MGASandbox.set_slack(model_copy, Orig.z, float(slack))
    start_time = time_ns()
    New_Point_List, New_z_list, new_p_list = MGASandbox.initialize_lists(Orig.point_list, Orig.z, iterations, Orig.nvars)
    current_points = fill(-1.0, (length(Orig.nvars),1))
    current_points[:,1] = New_Point_List'[:,1]
    init_time = time_ns() - start_time
    counter = 0
    current_point = fill(-1.0, length(Orig.nvars))
    avg_dist = 0
    prev_avg_dist = 0

    prev_10_dists = 0
    current_10_dists = 0
    conv_crit = 0.1
    conv = 1
    conv_vec= Vector{Float64}(undef,0)
    M = 100
    dists = fill(-1.0, iterations)
    @variable(model_copy, alpha >= 0)
    #@variable(model_copy, Djk[t=1:length(Orig.nvars),1:iterations+1]>=0)
    #@variable(model_copy, xplus[1:length(Orig.nvars)] >= 0)
    #@variable(model_copy, xminus[1:length(Orig.nvars)] >=0)
    #@variable(model_copy, cpplus[1:length(Orig.nvars), 1:iterations+1] >= 0)
    #@variable(model_copy, cpminus[1:length(Orig.nvars), 1:iterations+1] >= 0)
    #@variable(model_copy, z[1:length(Orig.nvars)], Bin)
    @variable(model_copy, s[t=1:length(Orig.nvars), k=1:iterations+1], Bin)
    @constraints(model_copy, begin
        #c_dist1[t=1:length(Orig.nvars)], model_copy[:x][t] == xplus[t]- xminus[t]
        #c_dist2a[t=1:length(Orig.nvars)], z[t] => {xplus[t] <= M}#Djk
        #c_dist2b[t=1:length(Orig.nvars)], !z[t] => {xplus[t] <= 0}
        #c_dist3a[t=1:length(Orig.nvars)], z[t] => {xminus[t] <= 0}
        #c_dist3b[t=1:length(Orig.nvars)], !z[t] => {xminus[t] <= M}
        #cp_dist1[t=1:length(Orig.nvars),k=1], current_points[t,k] == cpplus[t,k]- cpminus[t,k]
        #cp_dist2a[t=1:length(Orig.nvars),k=1:iterations+1], s[t,k] => {cpplus[t,k] <= M}#Djk
        #cp_dist2b[t=1:length(Orig.nvars),k=1:iterations+1], !s[t,k] => {cpplus[t,k] <= 0}
        #cp_dist3a[t=1:length(Orig.nvars),k=1:iterations+1], s[t,k] => {cpminus[t,k] <= 0}
        #cp_dist3b[t=1:length(Orig.nvars),k=1:iterations+1], !s[t,k] => {cpminus[t,k] <= M}
        cp_dist4[t=1:length(Orig.nvars),k=1], model_copy[:x][t] - current_points[t,k] + M*s[k] >= alpha
        cp_dist5[t=1:length(Orig.nvars),k=1], -model_copy[:x][t] + current_points[t,k] + M*(1-s[k]) >= alpha
        #c_alpha, alpha <= sum(Djk[t,1] for t in 1:length(Orig.nvars))
    end)
    @objective(model_copy, Max, alpha)
    init_time2 = time_ns()
    optimize!(model_copy)
    solve_time = time_ns() - init_time2
    New_Point_List, New_z_list, new_p_list, current_point = MGASandbox.update_iteration(New_Point_List, New_z_list, new_p_list, 1, model_copy, BA, N)
    current_points = hcat(current_points,New_Point_List'[:,counter+1])
    
    for i in 1:iterations
        prev_avg_dist = avg_dist
        counter = counter+ 1
        loop_time = time_ns()
        @constraint(model_copy, [t=1:length(Orig.nvars),k=counter+1], model_copy[:x][t] - current_points[t,k] + M*s[k] >= alpha)
        @constraint(model_copy, [t=1:length(Orig.nvars),k=counter+1], -model_copy[:x][t] + current_points[t,k] + M*(1-s[k]) >= alpha)

        #if counter == 1    
            #delete(model_copy, c_alpha)
            #unregister(model_copy, :c_alpha)
        #end
        #@constraint(model_copy, c_alpha, alpha <= sum(Djk[t,k] for t in 1:length(Orig.nvars), k in 1:counter+1))
        @objective(model_copy, Max, alpha)#sum(Djk[k] for k in 1:counter+1)   [t,k] for t in 1:length(Orig.nvars), k in 1:counter+1,sum(Djk[k] for k in 1:counter+1)
        obj_time = time_ns() - loop_time
        init_time2 = time_ns()
        optimize!(model_copy)
        solve_time = time_ns() - init_time2
        New_Point_List, New_z_list, new_p_list, current_point = MGASandbox.update_iteration(New_Point_List, New_z_list, new_p_list, i, model_copy, BA, N)
        current_points = hcat(current_points,New_Point_List'[:,counter+1])
        avg_dist = est_chull_vol(New_Point_List) #evaluate_manhattan_dist(current_point,New_Point_List)
        dists[i] = avg_dist

        if mod(i,10) == 0
            current_10_dists = rolling_avg_dist(dists,i)
            conv = abs(current_10_dists-prev_10_dists)
            push!(conv_vec,conv)
            if current_10_dists <= conv_crit
                break
            end

        end
        prev_10_dists = current_10_dists
    end
    tot_time = time_ns() - start_time
    times = MGASandbox.timing(init_time, obj_time, solve_time, tot_time, 0, counter)
    print_dists(dists,method, length(Orig.nvars), conv_vec)
    return New_z_list, New_Point_List, new_p_list, times 
end
"""



function timed_heuristic(Orig::MGASandbox.MGAInfo, iterations::Int64, slack::Float64, printer::Bool)
    solve_time = 0.0
    obj_time = 0.0
    method = 3
    vol_est_record = DataFrame(point_number = Int64[], actual = Float64[], estimate = Float64[])
    start_time = time_ns()
    New_Point_List, New_z_list, new_p_list = initialize_lists_Heur(Orig.point_list, Orig.z, iterations, Orig.nvars)
    init_time = time_ns() - start_time
    loop_time = time_ns()
    b = rand(-1:1,Int64(ceil(iterations/2)), length(Orig.nvars))#randn(Float64,(Int64(ceil(iterations/2)), length(Orig.nvars)))#
    obj_time = time_ns() - loop_time
    counter = 0
    current_point = fill(-1.0, length(Orig.nvars))
    avg_dist = 0
    prev_avg_dist = 0
    conv_vec= Vector{Float64}(undef,0)
    conv = 1
    conv_crit = 0.1
    dists = Vector{Float64}(undef,0)

    for i in 1:Int64(ceil(iterations/2))
        model_copy = copy(Orig.orig_model)
        set_silent(model_copy)
        set_optimizer(model_copy, CPLEX.Optimizer)
        model_copy = MGASandbox.set_slack(model_copy, Orig.z, float(slack))

        prev_avg_dist = avg_dist
        
        
        ismax = false
        @objective(model_copy, Min, sum(b[i,j]*all_variables(model_copy)[j] for j in eachindex(Orig.nvars)))
        init_time2 = time_ns()
        optimize!(model_copy)
        solve_time = time_ns() - init_time2
        current_point = value.(model_copy[:x])
        #println("At min")
        New_Point_List, New_z_list, new_p_list = update_iteration_heur(New_Point_List, New_z_list, new_p_list, i, model_copy, ismax, false)
        counter+=1
        if i > 2
            New_Point_List = unique(New_Point_List)
            avg_dist1 = est_chull_vol(New_Point_List) #evaluate_manhattan_dist(current_point,New_Point_List)
            #P1 = polyhedron(vrep(New_Point_List))
            #act = volume(P1)
            #if i>3 && act < vol_est_record[counter, 2]-1 
            #    println(P1)
            #end
            #push!(vol_est_record, [counter,act,avg_dist1])
            push!(dists,avg_dist1)
            
        end
        
        ismax = true
        @objective(model_copy, Max, sum(b[i,j]*all_variables(model_copy)[j] for j in eachindex(Orig.nvars)))
        optimize!(model_copy)
        New_Point_List, New_z_list, new_p_list= update_iteration_heur(New_Point_List, New_z_list, new_p_list, i, model_copy, ismax, false)
        #println("At max")
        counter+=1
        if i > 2
            New_Point_List = unique(New_Point_List)
            avg_dist1 = est_chull_vol(New_Point_List) #evaluate_manhattan_dist(current_point,New_Point_List)
            #P1 = polyhedron(vrep(New_Point_List))
            #act = volume(P1)
            ##if i>3 && act < vol_est_record[counter, 2]-1
            #    println(P1)
           # end
            #push!(vol_est_record, [counter,act,avg_dist1])
            push!(dists,avg_dist1)
            
        end
        #avg_dist2 = est_chull_vol(New_Point_List) #evaluate_manhattan_dist(current_point,New_Point_List)

        #push!(dists, avg_dist2)
        """
        if mod(2*i,10) == 0
            conv = abs(avg_dist-prev_avg_dist)
            push!(conv_vec,conv)
            println("conv is "*string(conv))
            if current_10_dists <= conv_crit
                break
            end

        end
        prev_10_dists = current_10_dists 
        """
    end
    finalvol = est_chull_vol(New_Point_List)
    #P2 =polyhedron(vrep(New_Point_List))
    #finalvol_act = volume(P2)
    #push!(vol_est_record, [0,finalvol_act,finalvol])
    push!(dists,finalvol)
    """
    ratio_dif = DataFrame(step = Int64[], actual_ratio = Float64[], estimate_diff = Float64[])
    for i in 4:nrow(vol_est_record)
        act_rat = vol_est_record[i,2]/vol_est_record[i-1,2]
        est_dif = vol_est_record[i,3]-vol_est_record[i-1,3]
        push!(ratio_dif, [i, act_rat,est_dif])
    end
    println(vol_est_record)
    println(ratio_dif)
    plt = plot(ratio_dif[!,:step],[ratio_dif[!,:actual_ratio],ratio_dif[!,:estimate_diff]])
    plt2=scatter(ratio_dif[!,:actual_ratio],ratio_dif[!,:estimate_diff])
    plt3=scatter(vol_est_record[!,2], vol_est_record[!,3])
    display(plt)
    display(plt3)
    """
    tot_time = time_ns() - start_time
    times = MGASandbox.timing(init_time, obj_time, solve_time, tot_time, 0, counter)

    println(vol_est_record)
    if printer == true
        print_dists(dists,method, length(Orig.nvars), conv_vec)
        return New_z_list, New_Point_List, new_p_list, times, b 
    else
        return New_z_list, New_Point_List, new_p_list, times, b, dists, conv_vec
    end
    
end


function timed_capacityminmax(Orig::MGASandbox.MGAInfo, iterations::Int64, slack::Float64, print::Bool)
    solve_time = 0.0
    obj_time = 0.0
    method = 6
    iterations = check_it_num(iterations, length(Orig.nvars))
    

    start_time = time_ns()
    New_Point_List, New_z_list, new_p_list = MGASandbox.initialize_lists(Orig.point_list, Orig.z, iterations, Orig.nvars)
    init_time = time_ns() - start_time
    loop_time = time_ns()
    
    
    a = Array{Int64,2}(undef,(0,length(Orig.nvars))) #randn(Float64,(Int64(ceil(iterations/2)), length(Orig.nvars)))#
    if length(Orig.nvars) == 3
        temp = permutations_with_replacement([-1,0,1],3)   
        a = [collect(perm) for perm in temp]  # Convert vectors to arrays
        a = transpose(reduce(hcat, a))  # Concatenate arrays into a single array
    else
        a = unique_int(rand(-1:1,2*iterations, length(Orig.nvars)))
        check_it_a!(a, iterations)
    end
    obj_time = time_ns() - loop_time
    counter = 0
    current_point = fill(-1.0, length(Orig.nvars))
    avg_dist = 0
    prev_avg_dist = 0
    conv_vec= Vector{Float64}(undef,0)
    conv = 1
    conv_crit = 0.1
    dists = Vector{Float64}(undef,0)

    Threads.@threads for i in 1:iterations
        model_copy = copy(Orig.orig_model)
        set_optimizer(model_copy, CPLEX.Optimizer)
        model_copy = MGASandbox.set_slack(model_copy, Orig.z, float(slack))
        prev_avg_dist = avg_dist
        counter += 1

        @objective(model_copy, Min, sum(a[i,j]*all_variables(model_copy)[j] for j in eachindex(Orig.nvars)))
        init_time2 = time_ns()
        optimize!(model_copy)
        solve_time = time_ns() - init_time2
        avg_dist = est_chull_vol(New_Point_List)#evaluate_manhattan_dist(current_point,New_Point_List) #estimate_vol(New_Point_List) #
        if i == 1
            conv_crit = avg_dist/50
        end 
        push!(dists,avg_dist)
        New_Point_List, New_z_list, new_p_list = MGASandbox.update_iteration(New_Point_List, New_z_list, new_p_list, i, model_copy, false, false)

    end
    tot_time = time_ns() - start_time
    times = MGASandbox.timing(init_time, obj_time, solve_time, tot_time, 0, counter)

    if print == true
        print_dists(dists,method, length(Orig.nvars), conv_vec)
        return New_z_list, New_Point_List, new_p_list, times, a 
    else
        return New_z_list, New_Point_List, new_p_list, times, a, dists, conv_vec
    end
    
end

function SPORES(Orig::MGASandbox.MGAInfo, iterations::Int64, slack::Float64)
    ender = 0
    BA = false
    N = false

    a=0.7
    b=1-a
    New_Point_List, New_z_list, new_p_list = initialize_lists(Orig.point_list, Orig.z, iterations, Orig.nvars)
    indic = fill(0, length(Orig.nvars))

    cap_mm=SPORES_obj(New_Point_List, Orig.nvars,iterations)
    println(cap_mm)
    println(Orig.nvars)
    for i in 1:iterations
        model_copy = copy(Orig.orig_model)
        set_optimizer(model_copy, CPLEX.Optimizer)
        model_copy = set_slack(model_copy, Orig.z, slack)
    
        indic = hsj_obj(New_Point_List, Orig.nvars, indic)
        println("indic is: "*string(indic))
        
        """
        if ender == 1
            break
        end
        if indic == ones(length(indic))
            ender = 1
        end
        """
        @expression(model_copy, MGA_Obj, a*sum(cap_mm[i,j]*all_variables(model_copy)[j] for j in eachindex(cap_mm[i])) + b*sum(indic[j]*all_variables(model_copy)[j] for j in eachindex(indic)))
        @objective(model_copy, Min, MGA_Obj)
        print(model_copy)
        init_time2 = time()
        optimize!(model_copy)
        solve_time = time() - init_time2

        z_temp = value(model_copy[:prev_obj])
        p_temp = value(model_copy[:MGA_Obj])
        #x_star_temp = value.(model_copy[i][:x])
        # println("Obj value: "*string(z_temp)*" x_opt: "*string(x_star_temp[1])*", "*string(x_star_temp[2])*", " *string(x_star_temp[3])*" p: "*string(p_temp))

        New_Point_List, New_z_list, new_p_list = update_iteration(New_Point_List, New_z_list, new_p_list, i, model_copy, BA, N)
    end

    return New_z_list, New_Point_List, new_p_list
end

function timed_SPORES(Orig::MGASandbox.MGAInfo, iterations::Int64, slack::Float64)
    ender = 0
    BA = false
    N = false
    solve_time = 0.0
    obj_time = 0.0
    method = 7

    a=0.9
    b=1-a
    dim = length(Orig.nvars)
    model_copy = copy(Orig.orig_model)
    set_silent(model_copy)
    set_optimizer(model_copy, CPLEX.Optimizer)
    model_copy = MGASandbox.set_slack(model_copy, Orig.z, slack)

    start_time = time_ns()
    New_Point_List, New_z_list, new_p_list = MGASandbox.initialize_lists(Orig.point_list, Orig.z, iterations, Orig.nvars)
    cap_mm=SPORES_obj(Orig.nvars,iterations)
    current_point = fill(-1.0, length(Orig.nvars))
    indic = fill(0, length(Orig.nvars))
    indic_list = fill(0, iterations, length(Orig.nvars))
    init_time = time_ns() - start_time
    counter = 0
    avg_dist = 0
    conv_vec= Vector{Float64}(undef,0)
    prev_avg_dist = 0
    conv = 1
    nuni = 0
    conv_crit = 0.1
    dists = fill(-1.0, iterations)
    
    for i in 1:iterations
        prev_avg_dist = avg_dist
        counter += 1
        loop_time = time_ns()
        indic = MGASandbox.hsj_obj(New_Point_List, Orig.nvars, indic)
        indic_list[i,:] = value.(-1*indic)
        println(value.(-1*indic))
        println(cap_mm[i])
        @objective(model_copy, Min, a*sum(cap_mm[i,j]*all_variables(model_copy)[j] for j in eachindex(cap_mm[i])) + b*sum(indic[j]*all_variables(model_copy)[j] for j in eachindex(indic)))
        obj_time = time_ns() - loop_time
        init_time2 = time_ns()
        optimize!(model_copy)
        println(value.(model_copy[:x]))
        solve_time = time_ns() - init_time2
        New_Point_List, New_z_list, new_p_list, current_point = MGASandbox.update_iteration(New_Point_List, New_z_list, new_p_list, i, model_copy, BA, N)
        avg_dist = est_chull_vol(New_Point_List) #evaluate_manhattan_dist(current_point,New_Point_List)
        dists[i] = avg_dist
        """
        if mod(i,10) == 0
            current_10_dists = rolling_avg_dist(dists,i)
            conv = abs(current_10_dists-prev_10_dists)
            println("Conv is "*string(conv))
            push!(conv_vec,conv)
            if current_10_dists <= conv_crit
                break
            end
        end
        
        prev_10_dists = current_10_dists
        """
    end
    tot_time = time_ns() - start_time
    times = MGASandbox.timing(init_time, obj_time, solve_time, tot_time, 0, counter)
    print_dists(dists,method, dim, conv_vec)
    New_Point_List, New_z_list, new_p_list,nuni = find_unique(New_Point_List, New_z_list, new_p_list)
    return New_z_list, New_Point_List, new_p_list, times 
end


function SPORES_obj(nvars::AbstractVector,iterations::Int64)
    a = unique_int(rand(-1:1,4*iterations, length(nvars)))
    check_it_a!(a, iterations)
    return a
end

function check_it_a!(a::AbstractArray, iterations::Int64)
    (r,c) = size(a)
    if iterations < r
        a = a[1:iterations,1:c]
        return a
    else
        println("Error")
    end
end

function permutations_with_replacement(arr::AbstractArray, k::Int64)
    if k == 1
        return [[x] for x in arr]
    else
        result = []
        sub_perms = permutations_with_replacement(arr, k - 1)
        for x in arr
            for sub_perm in sub_perms
                push!(result, [x, sub_perm...])
            end
        end
        return result
    end
end

function check_it_num(iterations::Int64, dimensions::Int64)
    println(dimensions)
    println(3^dimensions)
    println(iterations)
    if 3^dimensions < iterations
        iterations = 3^dimensions
    end
    println(iterations)
    return iterations
end

function elim_dupes(b::AbstractArray)
    r,c = size(b)
    count = 0
    for i in 1:r
        for j in 1:r
            if AngleBetweenVectors.angle(b[i,:],b[j,:]) < 0.01 && i != j
                b = b[begin:(j-1),(j+1):end]
                count+=1
            end
        end
    end
    b = [b;rand(count,c)]
    return b

end

function initialize_lists_Heur(point_list::AbstractArray, z::Float64, iterations::Int64, nvars::AbstractArray)
    New_Point_List = Array{Float64,2}(undef,(1,length(nvars)))
    New_Point_List[1,:] = point_list[1,:]

    New_z_list = fill(-1.0, 1)
    New_z_list[1] = z
    new_p_list = fill(-1.0, 1)
    new_p_list[1] = 0

    return New_Point_List, New_z_list, new_p_list
end

function update_iteration_heur(Point_list::AbstractArray, z_list::AbstractVector, p_list::AbstractVector, iter::Int64, model::Model, ismax::Bool, BA::Bool)
    """
    if ismax == false
        for i in eachindex(value.(model[:x]))
            Point_list[2*iter,i] = value.(model[:x])[i]
        end
        z_list[2*iter] = value(model[:prev_obj])
        if BA == false
            p_list[2*iter] = objective_value(model)
        else
            p_list[2*iter] = objective_value(model)
        end
    elseif ismax == true
        for i in eachindex(value.(model[:x]))
            Point_list[2*iter+1,i] = value.(model[:x])[i]
        end
        z_list[2*iter+1] = value(model[:prev_obj])
        p_list[2*iter+1] = objective_value(model)
    end
    """
    Point_list=[Point_list;value.(model[:x])']
    push!(z_list,value(model[:prev_obj]))
    push!(p_list,objective_value(model))

    return Point_list, z_list, p_list
end

function timed_MAA(Orig::MGASandbox.MGAInfo, iterations::Int64, slack::Float64, initial_it::Int64)
    solve_time = 0.0
    obj_time = Vector{Float64}(undef,0)
    obj_max = 0.0
    method = 4
    step_one_t = MGASandbox.timing(0.0, 0.0, 0.0, 0.0 , 0, 0)
    

    start_time = time_ns()
    point_list, z_list, p_list = initialize_lists_MAA(Orig, initial_it, iterations)
    init_time = time_ns() - start_time
    i = 0
    z_list, point_list, p_list, step_one_t, b, init_dists, init_conv = timed_heuristic(Orig, initial_it, float(slack), false) 
    prev_vecs = b
    runs = 0
    done = initial_it
    current_point = fill(-1.0, length(Orig.nvars))
    avg_dist = 0
    prev_avg_dist = 0
    conv = 1
    conv_crit = 0.1
    conv_vec= init_conv
    dists = init_dists
    ender = 0
    num_sols = 0
    prev_sols = 0
    while done < iterations
        count = 1
        println(iterations)
        println(done)
        loop_time = time_ns()
        point_list, z_list, p_list = MGASandbox.MAA_reformat(point_list,z_list, p_list)
        """
        if done > initial_it
            conv_crit = conv
            conv = length(z_list)    
            if conv == conv_crit
                ender = 1
                break
            end
            push!(conv_vec,conv)
            println("conv is "*string(conv))
        end
        """
        vecs = find_face_normals(polyhedron(vrep(point_list)))
        vecs, prev_vecs = discard_repeats(vecs, prev_vecs)
        push!(obj_time, time_ns() - loop_time)
        if maximum(obj_time) > 10^10
            println("Time out")
            ender = 1
            break
        end
        counter = 0
        Threads.@threads for i in eachindex(vecs)
            model_copy = copy(Orig.orig_model)
            set_silent(model_copy)
            set_optimizer(model_copy, CPLEX.Optimizer)
            model_copy = MGASandbox.set_slack(model_copy, Orig.z, float(slack)) # MGASandbox.BAMGAInit(Orig, iterations, initial_it, slack)
            prev_avg_dist = avg_dist
            counter = counter + 1
            runs = runs + 1
            @objective(model_copy, Max, sum(vecs[counter][j]*all_variables(model_copy)[j] for j in eachindex(Orig.nvars)))

            #model_copy, nset = MAASetup(model_copy,  Orig, point_list, z_list, p_list)
            
            init_time2 = time_ns()
            #set_start_value.(model_copy[:x], current_point)
            optimize!(model_copy)
            solve_time = time_ns() - init_time2
            current_point = value.(model_copy[:x])
            point_list, z_list, p_list = update_iteration_MAA(point_list, z_list, p_list, model_copy)
            avg_dist = est_chull_vol(point_list) #evaluate_manhattan_dist(current_point,New_Point_List)
            push!(dists, avg_dist)
            """
            if mod(runs,10) == 0
                current_10_dists = rolling_avg_dist(dists,runs)
                conv = abs(current_10_dists-prev_10_dists)
                push!(conv_vec,conv)
                println("conv is "*string(conv))
                
                if current_10_dists <= conv_crit
                    ender = 1
                    break
                end
                
            end
            prev_10_dists = current_10_dists
            """
            if runs >= iterations
                break
            end
            
        end
        #point_list, z_list, p_list, i, solve_time = MAASolver(model_copy, nset, iterations, done, point_list,z_list,p_list)
        done = done + counter
        if ender == 1
            break
        end
        (sols, vars) = size(point_list)
        num_sols = sols - prev_sols
        prev_sols = sols
        """
        if num_sols == 0
            break
        end
        """
    end
    obj_max = maximum(obj_time)
    tot_time = time_ns() - start_time
    times = MGASandbox.timing(init_time, obj_max, solve_time, tot_time, 0, done)
    print_dists(dists,method, length(Orig.nvars), conv_vec)
    return z_list, point_list, p_list, times
end
"""
function MAASetup(model_copy::AbstractVector, Orig::MGASandbox.MGAInfo, point_list::AbstractArray, z_list::AbstractArray, p_list::AbstractArray)
    points, z_list, p_list = MGASandbox.MAA_reformat(point_list,z_list, p_list)
    #nrows, ncols = size(points)
    vecs = find_face_normals(polyhedron(vrep(points)))
    counter = 0
    #mids = fill(-1.0,(nrows, ncols))
    #list = 1:nrows
    #point_ord = shuffle(collect(combinations(list,2)))
    
    
    for i in eachindex(vecs)
        counter = counter + 1
        #unregister(model_copy[i], :MGA_Obj_MAA)
        #@expression(model_copy[i], MGA_Obj_MAA, sum(vecs[counter][j]*all_variables(model_copy[i])[j] for j in eachindex(Orig.nvars)))
        println(size(all_variables(model_copy[i])))
        println(length(vecs))
        println(length(vecs[i]))
        @objective(model_copy[i], Max, sum(vecs[i][j]*all_variables(model_copy[1])[j] for j in eachindex(Orig.nvars)))
    end
    # println(mapreduce(permutedims, vcat, vecs))
    #plt = plot_objs(points, z_list,cent, mids, mapreduce(permutedims, vcat, vecs))
    #display(plt)
    return model_copy, counter
end

function MAASolver(model_copy::AbstractArray, counter::Int64, its::Int64, done::Int64, points::AbstractArray, z_list::AbstractArray, p_list::AbstractArray)
    remains = its - done
    count = 0
    solve_time = 0.0
    for i in 1:counter
        if i > remains
            break
        end
        set_optimizer(model_copy[i], CPLEX.Optimizer)
        init_time = time_ns()
        optimize!(model_copy[i])
        solve_time = time_ns() - init_time
        points, z_list, p_list = update_iteration_MAA(points, z_list, p_list,model_copy[i])
        count += 1
    end
    return points, z_list, p_list, count, solve_time
end
"""

function update_iteration_MAA(Point_list::AbstractArray, z_list::AbstractVector, p_list::AbstractVector,model::Model)
    Point_list = [Point_list;(value.(model[:x]))']
    z_list = push!(z_list,value(model[:prev_obj]))
    p_list= push!(p_list,objective_value(model))
    return Point_list, z_list, p_list
end

function find_face_normals(Poly::Polyhedra.VRep)
    hrep1 = hrep(Poly)
    hs = halfspaces(hrep1)
    vecs = Vector{Vector{Float64}}(undef, length(hs))
    iterator = 0
    for value in hs
        iterator += 1
        vecs[iterator] = value.a
    end
    return vecs
end

function initialize_lists_MAA(Orig::MGASandbox.MGAInfo, initial_it::Int64, iterations::Int64)
    point_list = fill(-1.0,(2*Int64(ceil(initial_it/2)) + iterations + 1, length(Orig.nvars)))
    point_list[1,:] = Orig.point_list[1,:]
    z_list = Vector{Float64}(undef, 2*Int64(ceil(initial_it/2)) + iterations + 1)
    z_list[1] = Orig.z
    p_list = Vector{Float64}(undef, 2*Int64(ceil(initial_it/2)) + iterations + 1)
    p_list[1] = 0
    return point_list, z_list, p_list
end

function discard_repeats(vecs::AbstractVector, prev_vecs::AbstractArray)
    nrow, ncol = size(prev_vecs)
    for i in vecs
        for j in 1:nrow
            if AngleBetweenVectors.angle(i,prev_vecs[j,:]) < 0.01
                deleteat!(vecs, findall(x->x==prev_vecs[j,:],vecs))
            else
                prev_vecs = [prev_vecs; i']
            end
        end
    end
    return vecs, prev_vecs
end

""" 
BA MGA Functions 

"""

function timed_BA(Orig::MGASandbox.MGAInfo, iterations::Int64, slack::Float64, initial_it::Int64)
    solve_time = 0.0
    obj_time = Vector{Float64}(undef,0)
    obj_max = 0.0
    step_one_t = MGASandbox.timing(0.0, 0.0, 0.0, 0.0 , 0, 0)
    b = Array{Float64,2}(undef,(Int64(ceil(initial_it/2)),length(Orig.nvars)))
    
    method = 5
    

    start_time = time_ns()
    point_list, z_list, p_list = initialize_lists_MAA(Orig, initial_it, iterations)
    i = 0
    z_list, point_list, p_list, step_one_t,b,init_dists,init_conv = timed_heuristic(Orig, initial_it, float(slack), false)
    prev_vecs = b
    dists = init_dists
    point_list, z_list, p_list = MGASandbox.MAA_reformat(point_list,z_list, p_list)
    runs = 0
    repeat = 0
    done = initial_it
    pairs = Vector{Vector{Int64}}(undef,0)
    opts = projection_options_gen(point_list)
    init_time = time_ns() - start_time
    current_point = fill(-1.0, length(Orig.nvars))
    avg_dist = 0
    prev_avg_dist = 0
    prev_10_dists = 0
    current_10_dists = 0
    conv_vec= init_conv
    conv = 1
    conv_crit = 0.3
    ender = 0
    num_sols = 0
    prev_sols = 0
    while done < iterations
        println(iterations)
        println(done)
        prev_avg_dist = avg_dist
        done_prev = done
        loop_time = time_ns()
        vecs, pairs = BAObjective(point_list, opts, pairs)
        vecs, prev_vecs = discard_repeats(vecs, prev_vecs)
        push!(obj_time, time_ns() - loop_time)
        if maximum(obj_time) > 10^10
            println("Time out")
            ender = 1
            break
        end
        counter = 0
        Threads.@threads for i in eachindex(vecs)
            model_copy = copy(Orig.orig_model)
            set_silent(model_copy)
            set_optimizer(model_copy, CPLEX.Optimizer)
            model_copy = MGASandbox.set_slack(model_copy, Orig.z, float(slack))
            counter = counter + 1
            runs = runs + 1
            @objective(model_copy, Max, sum(vecs[counter][j]*all_variables(model_copy)[j] for j in eachindex(Orig.nvars)))# Max
            init_time2 = time_ns()
            #set_start_value.(model_copy[:x], current_point)
            optimize!(model_copy)
            solve_time = time_ns() - init_time2
            current_point = value.(model_copy[:x])
            point_list, z_list, p_list = update_iteration_MAA(point_list, z_list, p_list, model_copy)
            avg_dist = est_chull_vol(point_list) #evaluate_manhattan_dist(current_point,New_Point_List)
            push!(dists,avg_dist)
            """
            if mod(runs,10) == 0
                current_10_dists = rolling_avg_dist(dists, runs)
                conv = abs(current_10_dists-prev_10_dists)
                push!(conv_vec,conv)
            end
            
            if current_10_dists <= conv_crit
                ender = 1
                break
            end
            """
            if runs >= iterations
                break
            end
            #prev_10_dists = current_10_dists
        end
        point_list, z_list, p_list = MGASandbox.MAA_reformat(point_list,z_list, p_list)
        """
        conv_crit = conv
        conv = length(z_list)    
        if conv == conv_crit
            ender = 1
            break
        end
        push!(conv_vec,conv)
        println("conv is "*string(conv))
        """
        done = done + counter
        if ender == 1
            break
        end
        (sols, vars) = size(point_list)
        num_sols = sols - prev_sols
        prev_sols = sols
        """
        if num_sols == 0
            break
        end
        """
    end
    obj_max = maximum(obj_time)
    tot_time = time_ns() - start_time
    times = MGASandbox.timing(init_time, obj_max, solve_time, tot_time, 0, done)
    print_dists(dists,method, length(Orig.nvars), conv_vec)
    println("BA Complete")
    return z_list, point_list, p_list, times
end

# Todos here:  parents are freshened each turn, indics are compiled, and children are updated
function projection_options_gen(point_list::AbstractArray)
    init_pointnum, ncol = size(point_list)
    init_points_ord = shuffle(collect(combinations(1:init_pointnum, 2)))
    init_mids = fill(-1.0, (length(init_points_ord), ncol))
    init_quarts = fill(-1.0, (2*length(init_points_ord), ncol))
    for i in eachindex(init_points_ord)
        init_mids[i,:] = 0.5*(point_list[init_points_ord[i][1],:] + point_list[init_points_ord[i][2],:])
        init_quarts[2*i-1,:] = 0.5*(point_list[init_points_ord[i][1],:] + init_mids[i,:])
        init_quarts[2*i,:] = 0.5*(point_list[init_points_ord[i][2],:] + init_mids[i,:])
    end
    opts = [init_mids;init_quarts]
    return opts
end

function BAObjective(point_list::AbstractArray, opts::AbstractArray, prev_pairs::AbstractVector)
    nrow, ncol = size(point_list)
    indicator = 0
    rows = 1:nrow
    point_order = shuffle(collect(combinations(rows, 2)))
    #prev_pairs = [prev_pairs;point_order]
    mids = fill(-1.0, (length(point_order), ncol))
    objs = Vector{Vector{Float64}}(undef, length(point_order))
    optnum, vars2 = size(opts)
    vecs = Vector{Vector{Float64}}(undef, optnum)
    mags = Vector{Float64}(undef, optnum)
    largest = fill(0, length(point_order))
    println(length(point_order))
    Threads.@threads for i in eachindex(point_order)
        mids[i,:] = 0.5*(point_list[point_order[i][1],:] + point_list[point_order[i][2],:])
    end
    Threads.@threads for i in eachindex(point_order)
        """
        for j in 1:optnum
            vecs[j] = mids[i,:] -vec(opts[j,:])
            mags[j] = dot(vecs[j],vecs[j])
            if mags[j] > largest[i]
                largest[i] = j
            end
        end
        """
        a = rand(1:optnum)
        if vec(opts[a,:]) in mids[i,:]
            a = mod(a,i) + 1
        end
        objs[i] = mids[i,:] -vec(opts[a,:]) # mids[a,:] point_list[1,:]-
    end
    b = float(rand(-5:5))
    Threads.@threads for j in 1:ncol
        for i in 1:length(objs)
            if  isapprox(objs[i][j],0) && indicator == 0
                objs[i][j] = b
                indicator = 1
            end
        end
        indicator = 0
    end
    return objs, point_order #prev_pairs
end


# End BA MGA Functions


function find_unique(points::AbstractArray,z::AbstractVector,p::AbstractVector)
    pointst = transpose(points)
    nrow, ncol = size(pointst)
    """
    for j in 1:ncol
        for i in 1:nrow
            if isnan(pointst[i,j])
                pointst[i,j] = 0.0
            end
        end
    end
    """
    uniques = fill(-1.0, (nrow, ncol))
    z_uni = Vector{Float64}(undef, ncol)
    p_uni = Vector{Float64}(undef, ncol)
    counter=0
    for i in 1:ncol
        for k in 1:ncol
            if isapprox(pointst[:,i],uniques[:,k],atol=0.1)
                break
            elseif k == ncol
                counter = counter + 1
                uniques[:,counter] = pointst[:,i]
                z_uni[counter] = z[i]
                p_uni[counter] = p[i]
            end
        end
    end
    uniques = uniques[1:end, 1:counter]
    z_uni = z_uni[1:counter]
    p_uni = p_uni[1:counter]
    uniquesT = transpose(uniques)
    return uniquesT, z_uni, p_uni, counter
end

"""
Run hsj takes the problem and iterations, then runs hsj with those characteristics, returns times.
"""

function runner(Orig_info::MGASandbox.MGAInfo, iterations::Int64, slack::Float64, opt::Int64)
    """
    nvars = 7
    # Settings
    iterations = 10
    slack = 0.1 # 0.1 = 10%
    """
    initial_it = 2

    if opt == 1
        z_list, point_list, p_list, times =  timed_hsj(Orig_info, iterations, float(slack))
    elseif opt == 2
        z_list, point_list, p_list, times =  timed_distance(Orig_info, iterations, float(slack))
    elseif opt == 3
        z_list, point_list, p_list, times, b =  timed_heuristic(Orig_info, iterations, float(slack), true)
    elseif opt == 4
        z_list, point_list, p_list, times =  timed_MAA(Orig_info, iterations-initial_it, float(slack), initial_it)
    elseif opt == 5
        z_list, point_list, p_list, times = timed_BA(Orig_info, iterations-initial_it, float(slack), initial_it)
    elseif opt == 6
        z_list, point_list, p_list, times, a = timed_capacityminmax(Orig_info, iterations, float(slack), true)
    elseif opt == 7
        z_list, point_list, p_list, times =  timed_SPORES(Orig_info, iterations, float(slack))
    else
        error("Method not implemented yet!")
    end
    uniques, z_uni, p_uni, num_uni = MGASandbox.find_unique(point_list, z_list, p_list)
    #println(point_list)
    uniquesT = transpose(uniques)
    println(uniquesT)
    """
    if length(Orig_info.nvars) <= 10
        poly = polyhedron(vrep(uniquesT))
        vol = Polyhedra.volume(poly)
        println("Volume is: "*string(vol))
    else
        vol = 0.0
    end
    """
    times.num_sols = length(z_uni)

    folder = "C:\\Users\\mike_\\Documents\\ZeroLab\\MGA-Methods"
    output = "Outputs"
    outpath = joinpath(folder, output)
    run = "Points_example_M_$opt.csv"
    overallpath = joinpath(outpath, run)
    uniques_df = DataFrame(uniquesT,:auto)
    CSV.write(overallpath, uniques_df)
    filename = "all_points_meth"*string(opt)*"dim"*string(length(Orig_info.nvars))*".csv"
    filepath = joinpath(outpath, filename)
    points_df = DataFrame(point_list,:auto)
    CSV.write(filepath, points_df)

    """println(times.num_sols)
    println("Initialization Time: "*string(times.initialize_t)*" ns")
    println("Objective Creation Time: "*string(times.obj_t)*" ns")
    println("Solve Time: "*string(times.solve_t)*" ns")
    println("Total Time: "*string(times.overall_t)*" ns")"""
    return times #, vol
end

"""
Setup_dim_hsj takes the dimension and runs per dimension, then runs the for loop setting up the hsj run every time. Outputs vector of times for each run/dim
"""


function setup_dim_all_method(dim::Int64, runs_per_dim::Int64, slack::Float64)
    times1 = Vector{MGASandbox.timing}(undef, runs_per_dim)
    times2 = Vector{MGASandbox.timing}(undef, runs_per_dim)
    times3 = Vector{MGASandbox.timing}(undef, runs_per_dim)
    times4 = Vector{MGASandbox.timing}(undef, runs_per_dim)
    times5 = Vector{MGASandbox.timing}(undef, runs_per_dim)
    times6 = Vector{MGASandbox.timing}(undef, runs_per_dim)
    times7 =Vector{MGASandbox.timing}(undef, runs_per_dim)
    iterations = dim + 15
     
    if iterations > 400
        iterations = 400
    end
    
    opt = [1,2,3,4,5,6,7]
    for i in 1:runs_per_dim
        model, vars = LP_N(dim)
        if dim == 3
            model, vars = LP_3D()
            slack = 3.0
        end
        point_list = fill(-1.0, (iterations, dim)) # dim
        z = objective_value(model)
        x_star = value.(all_variables(model))
        point_list[1,:] = x_star
        #println(point_list)
        Orig_info = MGASandbox.MGAInfo(model, point_list, z, vars)
        println("Running Method 1")
        times1[i] = runner(Orig_info, iterations, float(slack), opt[1])
        println("Running Method 2")
        times2[i] = MGASandbox.timing(0.0, 0.0,  0.0,  0.0,  0.0,  0.0)#runner(Orig_info, iterations, float(slack), opt[2])
        println("Running Method 3")
        times3[i] = runner(Orig_info, iterations, float(slack), opt[3])
        println("Running Method 4")
        if dim >= 0
            times4[i] = MGASandbox.timing(0.0, 0.0,  0.0,  0.0,  0.0,  0.0)
        else
            times4[i] = runner(Orig_info, iterations, float(slack), opt[4])
        end
        
        println("Running Method 5")
        if dim >= 0
            times5[i] = MGASandbox.timing(0.0, 0.0,  0.0,  0.0,  0.0,  0.0)
        else
            times5[i] = runner(Orig_info, iterations, float(slack), opt[5])
        end
        println("Running Method 6")
        times6[i] = runner(Orig_info, iterations, float(slack), opt[6])
        println("Running Method 7")
        times7[i] = runner(Orig_info, iterations, float(slack), opt[7])
    end
    
    return times1, times2, times3, times4, times5, times6, times7
end

function test_vol()
    dim = 6
    iterations=40
    slack=1.0
    model, vars = LP_N(dim)
    if dim == 3
        model, vars = LP_3D()
        slack = 3.0
    end
    point_list = fill(-1.0, (iterations, dim)) # dim
    z = objective_value(model)
    x_star = value.(all_variables(model))
    point_list[1,:] = x_star
    #println(point_list)
    Orig_info = MGASandbox.MGAInfo(model, point_list, z, vars)
    runner(Orig_info, iterations, float(slack), 3)
end

function many_runs_all(runs_per_dim::Int64, dims::Vector, slack::Float64)
    total_dims = length(dims)
    total_runs = runs_per_dim*total_dims
    all_times1 = Vector{AbstractVector}(undef, total_dims)
    all_times2 = Vector{AbstractVector}(undef, total_dims)
    all_times3 = Vector{AbstractVector}(undef, total_dims)
    all_times4 = Vector{AbstractVector}(undef, total_dims)
    all_times5 = Vector{AbstractVector}(undef, total_dims)
    all_times6 = Vector{AbstractVector}(undef, total_dims)
    all_times7 = Vector{AbstractVector}(undef, total_dims)
    for j in 1:length(dims)
        all_times1[j],all_times2[j],all_times3[j],all_times4[j],all_times5[j], all_times6[j], all_times7[j] = setup_dim_all_method(dims[j], runs_per_dim, float(slack))
    end
    return all_times1, all_times2,all_times3,all_times4,all_times5, all_times6, all_times7
end

"""
Unpack all times breaks down the vector of vector of times into readable stats in a DataFrame

This should be structured as follows

        | Dimension | Stats | Stats | Stats |
Row     |    3      | ..........
"""

function unpack_all_times(all_times::AbstractVector)
    #println(length(all_times))
    DFs = Vector{DataFrame}(undef, length(all_times))
    Descriptors = Vector{DataFrame}(undef, length(all_times))
    for j in 1:length(all_times)
        DFs[j] = times2df(all_times[j])
        Descriptors[j] = describe(DFs[j])
    end
    return DFs, Descriptors
end

function unique(points::AbstractArray)
    pointst = transpose(points)
    nrow, ncol = size(pointst)
    """
    for j in 1:ncol
        for i in 1:nrow
            if isnan(pointst[i,j])
                pointst[i,j] = 0.0
            end
        end
    end
    """
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
    return uniquesT
end

function unique_int(points::AbstractArray)
    pointst = transpose(points)
    nrow, ncol = size(pointst)
    """
    for j in 1:ncol
        for i in 1:nrow
            if isnan(pointst[i,j])
                pointst[i,j] = 0.0
            end
        end
    end
    """
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
    return uniquesT
end

function est_chull_vol(points::AbstractArray)
    point_copy = unique(points)
	(nit, nvars) = size(point_copy)
	pairs = collect(combinations(1:nvars, 2))
	pairwise_caps = fill(0.0, (nit,2))
	areas = Vector{Float64}(undef, 0)
	tot_area = 0.0
	for i in eachindex(pairs)
		pairwise_caps[:,1] = view(point_copy,:,pairs[i][1])
		pairwise_caps[:,2] = view(point_copy,:,pairs[i][2])
		uni_pc = unique(pairwise_caps)
		poly = polyhedron(vrep(Matrix(uni_pc)))
		vol = 0.0
		vol = Polyhedra.volume(poly)
        push!(areas, vol)
	end
	tot_area = sum(areas[i] for i in eachindex(areas))
	println("Volume is: "*string(tot_area))
	return tot_area
end


function summarize_times_all(Descriptors::AbstractVector, dims::AbstractVector)
    OverallDesc = DataFrame([[],[],[],[],[],[]], ["Init Time (ns)", "Objective Calc (ns)", "Solve Time (ns)","Overall Time (ns)" ,"Solutions Found", "Iterations Run"])
    for i in eachindex(Descriptors)
        means = Descriptors[i].mean
        push!(OverallDesc, means)
    end
    insertcols!(OverallDesc, 1, :Dimensions => dims)
    return OverallDesc
end

function times2df(t::AbstractVector{MGASandbox.timing})
    df = DataFrame()
    inits = fill(-1.0, length(t))
    objs = fill(-1.0, length(t))
    sols = fill(-1.0, length(t))
    overall = fill(-1.0, length(t))
    nsol = fill(-1.0, length(t))
    iterscomp = fill(-1.0, length(t))
    for i in eachindex(t)
        inits[i]= t[i].initialize_t
        objs[i] = t[i].obj_t
        sols[i] = t[i].solve_t
        overall[i] = t[i].overall_t
        nsol[i] = t[i].num_sols
        iterscomp[i] = t[i].iters_comp
    end
    df.Init_t = inits
    df.Obj_t = objs
    df.Sol_t = sols
    df.Overall_t = overall
    df.num_sol = nsol
    df.iters_comp = iterscomp
    return df
end

function print_dists(dists::AbstractVector, method::Int64, dim::Int64,conv::AbstractVector)
    folder = "C:\\Users\\mike_\\Documents\\ZeroLab\\MGA-Methods"
    output = "Outputs"
    namestr = "Method_"*string(method)*"_"*string(dim)*"Dimensions"
    file = "dists_"*namestr*".csv"
    convfile = "conv_"*namestr*".csv"
    outfold = joinpath(folder,output)
    path = joinpath(outfold,file)
    convpath = joinpath(outfold,convfile)
    
    distdf = DataFrame(Avg_Dists = dists)
    convdf = DataFrame(Conv = conv)
    CSV.write(path,distdf)
    CSV.write(convpath, convdf)
end



function printer_all(Dfs1::AbstractVector, Descriptors1::AbstractVector,Dfs2::AbstractVector, Descriptors2::AbstractVector, Dfs3::AbstractVector, Descriptors3::AbstractVector,Dfs4::AbstractVector, Descriptors4::AbstractVector,Dfs5::AbstractVector, Descriptors5::AbstractVector,Dfs6::AbstractVector, Descriptors6::AbstractVector,Dfs7::AbstractVector, Descriptors7::AbstractVector, Overall1_df::AbstractDataFrame, Overall2_df::AbstractDataFrame,Overall3_df::AbstractDataFrame, Overall4_df::AbstractDataFrame,Overall5_df::AbstractDataFrame, Overall6_df::AbstractDataFrame, Overall7_df::AbstractDataFrame,dims::AbstractVector)
    folder = "C:\\Users\\mike_\\Documents\\ZeroLab\\MGA-Methods"
    output = "Outputs"
    raw = "raw_times.csv"
    stats = "stats.csv"
    outpath = joinpath(folder,output)
    if isdir(outpath) == false
        mkdir(outpath)
    end
    methodstr = ["HSJ_MGA","Distance_MGA","Heuristic_MGA","MAA_MGA","BA_MGA","Cap_MinMax","SPORES"]
    casename = Vector{String}(undef,length(methodstr))
    df_case = Vector{String}(undef,length(methodstr))
    desc_case = Vector{String}(undef,length(methodstr))
    df_csv =  Vector{String}(undef,length(methodstr))
    desc_csv =  Vector{String}(undef,length(methodstr))
    for i in eachindex(Dfs1)
        for j in eachindex(methodstr)
            casename[j]= methodstr[j]*"_"*string(dims[i])*"_d_"
            df_case[j] = casename[j]*raw
            desc_case[j] = casename[j]*stats
            df_csv[j] = joinpath(outpath,df_case[j])
            desc_csv[j] = joinpath(outpath,desc_case[j])
        end
        CSV.write(df_csv[1], Dfs1[i])
        CSV.write(desc_csv[1], Descriptors1[i])
        CSV.write(df_csv[2], Dfs2[i])
        CSV.write(desc_csv[2], Descriptors2[i])
        CSV.write(df_csv[3], Dfs3[i])
        CSV.write(desc_csv[3], Descriptors3[i])
        CSV.write(df_csv[4], Dfs4[i])
        CSV.write(desc_csv[4], Descriptors4[i])
        CSV.write(df_csv[5], Dfs5[i])
        CSV.write(desc_csv[5], Descriptors5[i])
        CSV.write(df_csv[6], Dfs6[i])
        CSV.write(desc_csv[6], Descriptors6[i])
        CSV.write(df_csv[7], Dfs7[i])
        CSV.write(desc_csv[7], Descriptors7[i])
    end

    if length(dims) == 1
        overall1 = methodstr[1]*"_overall_"*string(dims[1])*stats
        overall2= methodstr[2]*"_overall_"*string(dims[1])*stats
        overall3 = methodstr[3]*"_overall_"*string(dims[1])*stats
        overall4= methodstr[4]*"_overall_"*string(dims[1])*stats
        overall5 = methodstr[5]*"_overall_"*string(dims[1])*stats
        overall6 = methodstr[6]*"_overall_"*string(dims[1])*stats
        overall7 = methodstr[7]*"_overall_"*string(dims[1])*stats
        overallpath1 = joinpath(outpath, overall1)
        CSV.write(overallpath1, Overall1_df)
        overallpath2 = joinpath(outpath, overall2)
        CSV.write(overallpath2, Overall2_df)
        overallpath3 = joinpath(outpath, overall3)
        CSV.write(overallpath3, Overall3_df)
        overallpath4 = joinpath(outpath, overall4)
        CSV.write(overallpath4, Overall4_df)
        overallpath5 = joinpath(outpath, overall5)
        CSV.write(overallpath5, Overall5_df)
        overallpath6 = joinpath(outpath, overall6)
        CSV.write(overallpath6, Overall6_df)
        overallpath7 = joinpath(outpath, overall7)
        CSV.write(overallpath7, Overall7_df)
    end

end

"""
Main holds all settings, runs the overall process, and prints
"""

function main()
    slack = 0.1
    runs_per_dim = 10 
    dims = [3,5,10,20,30,50]

    all_times1, all_times2, all_times3, all_times4, all_times5, all_times6,all_times7 = many_runs_all(runs_per_dim, dims, float(slack))
    DFs1, Descriptors1 = unpack_all_times(all_times1)
    DFs2, Descriptors2 = unpack_all_times(all_times2)
    DFs3, Descriptors3 = unpack_all_times(all_times3)
    DFs4, Descriptors4 = unpack_all_times(all_times4)
    DFs5, Descriptors5 = unpack_all_times(all_times5)
    DFs6, Descriptors6 = unpack_all_times(all_times6)
    DFs7, Descriptors7 = unpack_all_times(all_times7)

    OverallDesc1 = summarize_times_all(Descriptors1, dims)
    OverallDesc2 = summarize_times_all(Descriptors2, dims)
    OverallDesc3 = summarize_times_all(Descriptors3, dims)
    OverallDesc4 = summarize_times_all(Descriptors4,dims)
    OverallDesc5 = summarize_times_all(Descriptors5,dims)
    OverallDesc6 = summarize_times_all(Descriptors6,dims)
    OverallDesc7 = summarize_times_all(Descriptors7,dims)
    for i in eachindex(DFs1)
        println("Results for "*string(dims[i])*" dimensions")
        println("Method: HSJ")
        println(DFs1[i])
        println(Descriptors1[i])
        println("Results for "*string(dims[i])*" dimensions")
        println("Method: Distance")
        println(DFs2[i])
        println(Descriptors2[i])
        println("Results for "*string(dims[i])*" dimensions")
        println("Method: Heuristic")
        println(DFs3[i])
        println(Descriptors3[i])
        println("Results for "*string(dims[i])*" dimensions")
        println("Method: MAA")
        println(DFs4[i])
        println(Descriptors4[i])
        println("Results for "*string(dims[i])*" dimensions")
        println("Method: BA MGA")
        println(DFs5[i])
        println(Descriptors5[i])
        println("Results for "*string(dims[i])*" dimensions")
        println("Method: Capacity Min/Max")
        println(DFs6[i])
        println(Descriptors6[i])
        println("Results for "*string(dims[i])*" dimensions")
        println("Method: SPORES")
        println(DFs7[i])
        println(Descriptors7[i])
    end
    println(OverallDesc1)
    println(OverallDesc2)
    println(OverallDesc3)
    println(OverallDesc4)
    println(OverallDesc5)
    println(OverallDesc6)
    println(OverallDesc7)
    printer_all(DFs1, Descriptors1,DFs2, Descriptors2, DFs3, Descriptors3,DFs4, Descriptors4,DFs5, Descriptors5,DFs6, Descriptors6, DFs7, Descriptors7,OverallDesc1, OverallDesc2,OverallDesc3, OverallDesc4,OverallDesc5,OverallDesc6,OverallDesc7, dims)
end



#old_main()
main()
# @btime MGASandbox.main()s
#test_vol()