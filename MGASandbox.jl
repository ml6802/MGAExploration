"""
Sandbox for MGA Methods Testing
"""
module MGASandbox

using JuMP, CPLEX, DataFrames, BenchmarkTools, Base.Threads, LinearAlgebra, Ipopt, PlotlyJS, Combinatorics, Random, QHull, Polyhedra, CDDLib, SimpleGraphs

"""
Original LP Problem for Sandbox
"""

function LP_3D()
    model = Model()
    @variable(model, x[1:3] >= 0)
    @constraints(model, begin
        c1, x[1] + x[2] + x[3] >= 2
        c2, x[1] <= 3
        c3, 2*x[2] + 3*x[3] <= 5
    end)
    println("Model created")
    nvars = all_variables(model) # Modularize
    println(nvars)
    return model, nvars
end

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

function solve_orig(model::Model)
    @expression(model, obj, model[:x][1] + 2*model[:x][2] + 2*model[:x][3]) # needs to be automated and modularized  model[:revenue] - model[:costs]
    @objective(model, Min, obj)
    print(model)
    set_optimizer(model, CPLEX.Optimizer)
    optimize!(model)
    z = value.(model[:obj])
    x_star = value.(all_variables(model))
    println("Obj value: "*string(z)*" x_star: "*string(x_star[1])*", "*string(x_star[2])*", "*string(x_star[3]))
    return z, x_star
end

function solve_orig_obj2(model::Model)
    @expression(model, obj, 3*model[:x][1] + model[:x][2] + model[:x][3]) # needs to be automated and modularized  model[:revenue] - model[:costs]
    @objective(model, Min, obj)
    print(model)
    set_optimizer(model, CPLEX.Optimizer)
    optimize!(model)
    z = value.(model[:obj])
    x_star = value.(all_variables(model))
    println("Obj value: "*string(z)*" x_star: "*string(x_star[1])*", "*string(x_star[2])*", "*string(x_star[3]))
    return z, x_star
end

mutable struct timing
    initialize_t::Float64
    obj_t::Float64
    solve_t::Float64
    overall_t::Float64
    num_sols::Int64
    iters_comp::Int64
end
struct MGAInfo
    orig_model::Model
    point_list::Array{Float64,2}
    z::Float64
    nvars::Array{VariableRef,1} ### May have to change for larger models
end

function update_iteration(Point_list::AbstractArray, z_list::AbstractVector, p_list::AbstractVector, iter::Int64, model::Model, BA::Bool, N::Bool)
    Point_list = [Point_list;value.(model[:x]')]
    current_point = value.(model[:x])
    push!(z_list,value(model[:prev_obj]))
    push!(p_list, objective_value(model))
    return Point_list, z_list, p_list, current_point
end

function initialize_lists(point_list::AbstractArray, z::Float64, iterations::Int64, nvars::AbstractArray)
    New_Point_List = fill(-1.0,(1, length(nvars)))
    New_Point_List[1,:] = point_list[1,:]

    New_z_list = fill(-1.0,1)
    New_z_list[1] = z
    new_p_list = fill(-1.0, 1)
    new_p_list[1] = 0

    return New_Point_List, New_z_list, new_p_list
end

function set_slack(model::Model, z::Float64, slack::Float64)
    @expression(model, prev_obj, objective_function(model)) # Currently manual, try to figure out a way to fix that  model[:revenue] - model[:costs]
    @constraint(model, c_slack, prev_obj <= z*(1+slack))
    return model
end

"""
Hop-Skip-Jump MGA related Functions
"""

function hsj(Orig::MGAInfo, iterations::Int64, slack::Float64)
    ender = 0
    BA = false
    N = false

    New_Point_List, New_z_list, new_p_list = initialize_lists(Orig.point_list, Orig.z, iterations, Orig.nvars)
    model_copy = Vector{Model}(undef, iterations)
    indic = fill(0, length(Orig.nvars))

    
    for i in 1:iterations
        model_copy[i] = copy(Orig.orig_model)
        set_optimizer(model_copy[i], CPLEX.Optimizer)
        model_copy[i] = set_slack(model_copy[i], Orig.z, slack)
    
        indic = hsj_obj(New_Point_List, Orig.nvars, indic)
        println("indic is: "*string(indic))

        if ender == 1
            break
        end
        if indic == ones(length(indic))
            ender = 1
        end
        @expression(model_copy[i], MGA_Obj, sum(indic[j]*all_variables(model_copy[i])[j] for j in eachindex(indic)))
        @objective(model_copy[i], Min, MGA_Obj)
        # print(model_copy[i])
        init_time2 = time()
        optimize!(model_copy[i])
        solve_time = time() - init_time2

        z_temp = value(model_copy[i][:prev_obj])
        p_temp = value(model_copy[i][:MGA_Obj])
        #x_star_temp = value.(model_copy[i][:x])
        # println("Obj value: "*string(z_temp)*" x_opt: "*string(x_star_temp[1])*", "*string(x_star_temp[2])*", " *string(x_star_temp[3])*" p: "*string(p_temp))

        New_Point_List, New_z_list, new_p_list = update_iteration(New_Point_List, New_z_list, new_p_list, i, model_copy[i], BA, N)
    end

    return New_z_list, New_Point_List, new_p_list
end

function SPORES(Orig::MGAInfo, iterations::Int64, slack::Float64)
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

function hsj_obj(point_list::AbstractArray, nvars::AbstractArray, indic::AbstractVector)
    nrow, ncol = size(point_list)
    indic = fill(0, ncol)
    for j in 1:ncol
        for k in 1:nrow
            if point_list[k, j] >= 0.01
                indic[j] += 1
            end
        end
    end
    return indic
end

function SPORES_obj(point_list::AbstractArray, nvars::AbstractArray,iterations::Int64)
    a = unique_int(rand(-1:1,2*iterations, length(nvars)))
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


"""
Distance Based MGA
"""

function Distance(Orig::MGAInfo, iterations::Int64, slack::Float64)
    ender = 0
    BA = false
    N = false

    New_Point_List, New_z_list, new_p_list = initialize_lists(Orig.point_list, Orig.z, iterations, Orig.nvars)
    model_copy = Vector{Model}(undef, iterations)

    for i in 1:iterations
        model_copy[i] = copy(Orig.orig_model)
        set_optimizer(model_copy[i], Ipopt.Optimizer)
        model_copy[i] = set_slack(model_copy[i], Orig.z, slack)

        if ender == 1
            break
        end

        @NLexpression(model_copy[i], MGA_Obj, sum(norm(sum(all_variables(model_copy[i])[j] - New_Point_List[k,j] for j in eachindex(Orig.nvars)), 1) for k in 1:i))
        @NLobjective(model_copy[i], Max, MGA_Obj)
        # print(model_copy[i])
        optimize!(model_copy[i])

        z_temp = value(model_copy[i][:prev_obj])
        p_temp = value(model_copy[i][:MGA_Obj])
        #x_star_temp = value.(model_copy[i][:x])
        #println("Obj value: "*string(z_temp)*" x_opt: "*string(x_star_temp[1])*", "*string(x_star_temp[2])*", " *string(x_star_temp[3])*" p: "*string(p_temp))

        New_Point_List, New_z_list, new_p_list = update_iteration(New_Point_List, New_z_list, new_p_list, i, model_copy[i], BA, N)
    end
    return New_z_list, New_Point_List, new_p_list
end

"""
Heuristic MGA Functions
"""

function Heuristic_MGA(Orig_Info::MGAInfo, iterations::Int64, slack::Float64)
    model_copy = Vector{Model}(undef, iterations)
    model_copy_max = Vector{Model}(undef, iterations)
    New_Point_List, New_z_List, new_p_list = initialize_lists_Heur(Orig_Info.point_list, Orig_Info.z, iterations, Orig_Info.nvars)

    model_copy, model_copy_max = HeuristicMGASetup(Orig_Info.orig_model, Orig_Info.z, iterations, Orig_Info.nvars, slack)
    New_Point_List, New_z_List, new_p_list = solve_Heuristic_seq(model_copy, model_copy_max, New_Point_List, New_z_List, new_p_list)

    return New_Point_List, New_z_List, new_p_list
end

function HeuristicMGASetup(model::Model, z::Float64, iterations::Int64, nvars::AbstractArray, slack::Float64)
    model_copy = Vector{Model}(undef, iterations)
    model_copy_max = Vector{Model}(undef, iterations)
    b = rand(iterations, length(nvars))
    for i in 1:iterations #  Threads.@threads
        model_copy[i] = copy(model)
        model_copy[i] = set_slack(model_copy[i], z, slack)
        model_copy_max[i] = copy(model_copy[i])
        set_optimizer(model_copy[i], CPLEX.Optimizer)
        set_optimizer(model_copy_max[i], CPLEX.Optimizer)
        @expression(model_copy[i], MGA_Obj, sum(b[i,j]*all_variables(model_copy[i])[j] for j in eachindex(nvars)))
        @expression(model_copy_max[i], MGA_Obj_max, sum(b[i,j]*all_variables(model_copy_max[i])[j] for j in eachindex(nvars)))
        @objective(model_copy[i], Min, MGA_Obj)
        @objective(model_copy_max[i], Max, MGA_Obj_max)
    end
    return model_copy, model_copy_max
end

function update_iteration_heur(Point_list::AbstractArray, z_list::AbstractVector, p_list::AbstractVector, iter::Int64, model::Model, ismax::Bool, BA::Bool)
    println(Point_list)
    (rows, cols) = size(Point_list)
    current_point = fill(-1.0, cols)
    Point_list=[Point_list;value.(model[:x])']
    push!(z_list,value(model[:prev_obj]))
    push!(p_list,objective_value(model))
    if ismax == true
        current_point = value.(model[:x])
    end

    return Point_list, z_list, p_list, current_point
end

"""
module Heuristic_par
    function solve_Heuristic_Parallel(model_copy::AbstractArray, model_copy_max::AbstractArray, z::Float64, point_list::AbstractArray)
        New_Point_List, New_z_list, new_p_list = initialize_lists_Heur(point_list, z, iterations, nvars)
    end
end
"""

function solve_Heuristic_seq(model_copy::AbstractArray, model_copy_max::AbstractArray, New_Point_List::AbstractArray, New_z_list::AbstractArray, new_p_list::AbstractArray)    
    BA = false
    for i in eachindex(model_copy) # Threads.@threads 
        print(model_copy[i])
        optimize!(model_copy[i])

        ismax = false
        z_temp = value(model_copy[i][:prev_obj])
        p_temp = value(model_copy[i][:MGA_Obj])
        #x_star_temp = value.(model_copy[i][:x])
        #println("Obj value: "*string(z_temp)*" x_opt: "*string(x_star_temp[1])*", "*string(x_star_temp[2])*", " *string(x_star_temp[3])*" p: "*string(p_temp))
        New_Point_List, New_z_list, new_p_list = update_iteration_heur(New_Point_List, New_z_list, new_p_list, i, model_copy[i], ismax, BA)

        print(model_copy_max[i])
        optimize!(model_copy_max[i])

        ismax = true
        z_temp_max = value(model_copy_max[i][:prev_obj])
        p_temp_max = value(model_copy_max[i][:MGA_Obj_max])
        #x_star_temp_max = value.(model_copy_max[i][:x])
        #println("Obj value: "*string(z_temp_max)*" x_opt: "*string(x_star_temp_max[1])*", "*string(x_star_temp_max[2])*", " *string(x_star_temp_max[3])*" p: "*string(p_temp_max))
        New_Point_List, New_z_list, new_p_list = update_iteration_heur(New_Point_List, New_z_list, new_p_list, i, model_copy_max[i], ismax, BA)
    end
    return New_Point_List, New_z_list, new_p_list
end

function initialize_lists_Heur(point_list::AbstractArray, z::Float64, iterations::Int64, nvars::AbstractArray)
    New_Point_List = Array{Float64, 2}(undef, (0, length(nvars)))
    New_Point_List = [New_Point_List;point_list[1,:]']

    New_z_list = Vector{Float64}(undef, 2*iterations + 1)
    New_z_list[1] = z
    new_p_list = Vector{Float64}(undef, 2*iterations + 1)
    new_p_list[1] = 0

    return New_Point_List, New_z_list, new_p_list
end

"""
Modeling All Alternatives
"""
function MAA(Orig::MGAInfo, iterations::Int64, slack::Float64, initial_its::Int64)
    models = BAMGAInit(Orig, iterations, initial_its, slack)
    point_list = fill(-1.0,(initial_its + iterations + 1, length(Orig.nvars)))
    point_list[1,:] = Orig.point_list[1,:]
    z_list = Vector{Float64}(undef, 2*iterations + 1)
    z_list[1] = Orig.z
    p_list = Vector{Float64}(undef, 2*iterations + 1)
    p_list[1] = 0
    point_list, z_list, p_list = BAMGAStepOne(models, initial_its, point_list, z_list, p_list)
    point_list, z_list, p_list = MAA_StepN(models, iterations, initial_its, Orig, point_list, z_list, p_list)
    return point_list, z_list, p_list
end

function MAA_StepN(model_copy::AbstractArray, iterations::Int64, initial_its::Int64, Orig::MGAInfo, point_list::AbstractArray, z_list::AbstractArray, p_list::AbstractArray)
    done = initial_its
    while done < iterations
        model_copy, nset = MAASetup(model_copy,  Orig, point_list, z_list, p_list)
        point_list, z_list, p_list, i = MAASolver(model_copy, nset, iterations, done, point_list,z_list,p_list)
        done += i
    end
    return point_list, z_list, p_list
end

function MAA_reformat(points::AbstractArray, z::AbstractArray, p::AbstractArray)
    nrow, ncol = size(points)
    new_points = fill(-1.0, nrow)
    new_points, z, p, count = find_unique(points,z, p)
    new_pointsT = Matrix(transpose(new_points))
    return new_pointsT, z, p
end

function MAASetup(model_copy::AbstractArray, Orig::MGAInfo, point_list::AbstractArray, z_list::AbstractArray, p_list::AbstractArray)
    points, z_list, p_list = MAA_reformat(point_list,z_list, p_list)
    nrows, ncols = size(points)
    vecs = find_face_normals(polyhedron(vrep(points)))
    counter = 0
    cent = [1.66, 1.0, 0.0]
    mids = fill(-1.0,(nrows, ncols))
    list = 1:nrows
    point_ord = shuffle(collect(combinations(list,2)))
    for j in 1:nrows
        mids[j,:] = 0.5*(points[point_ord[j][1],:] + points[point_ord[j][2],:])
    end
    for i in eachindex(vecs)
        counter = counter + 1
        unregister(model_copy[i], :MGA_Obj_MAA)
        @expression(model_copy[i], MGA_Obj_MAA, sum(vecs[counter][j]*all_variables(model_copy[i])[j] for j in eachindex(Orig.nvars)))
        @objective(model_copy[i], Max, MGA_Obj_MAA)
    end
    println(mapreduce(permutedims, vcat, vecs))
    #plt = plot_objs(points, z_list,cent, mids, mapreduce(permutedims, vcat, vecs))
    #display(plt)
    return model_copy, counter
end

function MAASolver(model_copy::AbstractArray, counter::Int64, its::Int64, done::Int64, points::AbstractArray, z_list::AbstractArray, p_list::AbstractArray)
    remains = its - done
    count = 0
    for i in 1:counter
        if i > remains
            break
        end
        set_optimizer(model_copy[i], CPLEX.Optimizer)
        optimize!(model_copy[i])
        points, z_list, p_list = update_iteration_MAA(points, z_list, p_list, i, model_copy[i], done)
        count += 1
    end
    return points, z_list, p_list, count
end

function update_iteration_MAA(Point_list::AbstractArray, z_list::AbstractVector, p_list::AbstractVector, iter::Int64, model::Model, initial::Int64)
    for i in eachindex(value.(model[:x]))
        Point_list[iter + initial + 1,i] = value.(model[:x])[i]
    end
    z_list[iter + initial + 1] = value(model[:prev_obj])
    p_list[iter + initial + 1] = value(model[:MGA_Obj_MAA])
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


"""
Bow and Arrow MGA
Update with with new ideas soon 
"""
function BAMGA(Orig::MGAInfo, iterations::Int64, slack::Float64, initial_its::Int64)
    models = BAMGAInit(Orig, iterations, initial_its, slack)
    # point_list, new_z, new_p = initialize_lists_Heur(Orig.point_list, Orig.z, iterations, Orig.nvars)
    point_list = fill(-1.0,(initial_its + iterations + 1, length(Orig.nvars)))
    point_list[1,:] = Orig.point_list[1,:]

    z_list = Vector{Float64}(undef, initial_its + iterations + 1)
    z_list[1] = Orig.z
    p_list = Vector{Float64}(undef, initial_its + iterations + 1)
    p_list[1] = 0.0
    
    point_list, z_list, p_list = BAMGAStepOne(models, initial_its, point_list, z_list, p_list)
    point_list, z_list, p_list, cent = BAMGASetup(point_list, z_list, p_list)
    point_list, z_list, p_list = BAMGAStepN(models, iterations, initial_its, Orig, point_list, z_list, p_list, cent)
    
    return point_list, z_list, p_list
end

function BAMGASetup(point_list::AbstractArray, z_list::AbstractArray, p::AbstractArray)
    unique_points, z_list, p, counter = find_unique(point_list,z_list,p)
    unique_points = Matrix(transpose(unique_points))
    nrow, ncol = size(unique_points)
    cent = fill(-1.0, ncol)
    # cent = hchebyshevcenter(hrep(polyhedron(vrep(unique_points))), CPLEX.Optimizer)
    for i in 1:ncol
        avg = sum(unique_points[j, i] for j in 1:nrow)/nrow
        cent[i] = avg
    end
    #plt = plot_space(unique_points', z_list)
    #display(plt)
    return unique_points, z_list, p, cent
end

function BAMGAInit(Orig::MGAInfo, iterations::Int64, initial_its::Int64, slack::Float64)
    model_copy = Vector{Model}(undef, iterations + initial_its)
    for i in 1:iterations
        model_copy[i] = copy(Orig.orig_model)
        model_copy[i] = set_slack(model_copy[i], Orig.z, slack)
        set_optimizer(model_copy[i], CPLEX.Optimizer)
    end
    return model_copy
end

function BAMGAStepOne(model_copy::AbstractArray, initial_its::Int64, point_list::AbstractArray, z_list::AbstractArray, p_list::AbstractArray)
    nvars = num_variables(model_copy[1])
    model_copy_max = Vector{Model}(undef, initial_its)
    b = rand(initial_its, nvars)
    
    for i in 1:initial_its
        model_copy_max[i] = copy(model_copy[i])
        set_optimizer(model_copy[i], CPLEX.Optimizer)
        set_optimizer(model_copy_max[i], CPLEX.Optimizer)
        @expression(model_copy[i], MGA_Obj_rand, sum(b[i,j]*all_variables(model_copy[i])[j] for j in eachindex(nvars)))
        @expression(model_copy_max[i], MGA_Obj_max, sum(b[i,j]*all_variables(model_copy_max[i])[j] for j in eachindex(nvars)))
        @objective(model_copy[i], Min, MGA_Obj_rand)
        @objective(model_copy_max[i], Max, MGA_Obj_max)
    end
    point_list, z_list, p_list = solve_BA_StepOne(model_copy, model_copy_max, point_list, z_list, p_list, initial_its)
    return point_list, z_list, p_list
end

function solve_BA_StepOne(model_copy::AbstractArray, model_copy_max::AbstractArray, New_Point_List::AbstractArray, New_z_list::AbstractArray, new_p_list::AbstractArray, its::Int64)    
    BA = true
    for i in 1:its # Threads.@threads 
        # print(model_copy[i])
        optimize!(model_copy[i])

        ismax = false
        z_temp = value(model_copy[i][:prev_obj])
        p_temp = value(model_copy[i][:MGA_Obj_rand])
        #println("Objective value: "*string(z_temp)*" P value: "*string(p_temp))
        #x_star_temp = value.(model_copy[i][:x])
        #println("Obj value: "*string(z_temp)*" x_opt: "*string(x_star_temp[1])*", "*string(x_star_temp[2])*", " *string(x_star_temp[3])*" p: "*string(p_temp))
        New_Point_List, New_z_list, new_p_list = update_iteration_heur(New_Point_List, New_z_list, new_p_list, i, model_copy[i], ismax, BA)

        # print(model_copy_max[i])
        optimize!(model_copy_max[i])

        ismax = true
        #z_temp_max = value(model_copy_max[i][:prev_obj])
        #p_temp_max = value(model_copy_max[i][:MGA_Obj_max])
        #x_star_temp_max = value.(model_copy_max[i][:x])
        #println("Obj value: "*string(z_temp_max)*" x_opt: "*string(x_star_temp_max[1])*", "*string(x_star_temp_max[2])*", " *string(x_star_temp_max[3])*" p: "*string(p_temp_max))
        New_Point_List, New_z_list, new_p_list = update_iteration_heur(New_Point_List, New_z_list, new_p_list, i, model_copy_max[i], ismax, BA)
    end
    return New_Point_List, New_z_list, new_p_list
end

# Todos here:  parents are freshened each turn, indics are compiled, and children are updated

function BAMGAStepN(model_copy::AbstractArray, iterations::Int64, initial_its::Int64, Orig::MGAInfo, point_list::AbstractArray, z_list::AbstractArray, p_list::AbstractArray, cent::AbstractArray)
    BA = true
    N = true
    it_remain = iterations - initial_its
    buffer = 2*initial_its+1
    counter = 0
    local unis, uni_z, uni_p, num = find_unique(point_list, z_list,p_list)
    local unis = Matrix(transpose(unis))
    local initial_points = unis
    nrow, ncol = size(unis)
    in_pointsT = Matrix(transpose(initial_points))
    numbering = Vector(1:nrow)
    # Labelling orig points
    in_pointsT = [in_pointsT;numbering']
    in_points = Matrix(transpose(in_pointsT))


    local comp_point_list = unis
    local comp_z_list = uni_z
    local comp_p_list = uni_p
    local n_its = 0
    local counter = 0
    local indicators = fill(0, (length(uni_z), 2))

    
    local objs = Array{Float64,2}(undef, (nrow, ncol))
    local child_Point_List = fill(-1.0,(n_its, length(Orig.nvars)))
    local child_z_list = Vector{Float64}(undef, n_its)
    local child_p_list = Vector{Float64}(undef, n_its)
    local parent_Point_List = fill(-1.0,(n_its, length(Orig.nvars)))
    local parent_z_list = Vector{Float64}(undef, n_its)
    local parent_p_list = Vector{Float64}(undef, n_its)

    for i in 1:it_remain
        nrow, ncol = size(unis)
        obj_num, varnum = size(objs)
        if mod(i, obj_num) == 0 || i == 1
            parent_Point_List = unis
            parent_z_list = uni_z
            parent_p_list = uni_p
            #println("Objs were: $objs")
            nrow += n_its
            objs, used_parents, endpoints = BAObj(in_points, parent_Point_List,parent_z_list, parent_p_list, cent, indicators)
            n_its, r = size(objs)
            
            #println(used_parents)
            #println(endpoints)

            child_Point_List = fill(-1.0,(0, length(Orig.nvars)))
            child_z_list = Vector{Float64}(undef, 0)
            child_p_list = Vector{Float64}(undef, 0)
            counter = 0
        end
        counter = counter + 1
        #println("Objs are: $objs")
        @expression(model_copy[i], MGA_Obj_BA, sum(objs[counter,j]*all_variables(model_copy[i])[j] for j in eachindex(Orig.nvars)))
        @objective(model_copy[i], Max, MGA_Obj_BA)
        optimize!(model_copy[i])
        child_Point_List = [child_Point_List;value.(model_copy[counter][:x])']
        child_z_list = [child_z_list;value(model_copy[counter][:prev_obj])]
        child_p_list = [child_p_list;value(model_copy[counter][:MGA_Obj_BA])]
        z_temp = value(model_copy[counter][:prev_obj])
        p_temp = value(model_copy[counter][:MGA_Obj_BA])
        x_star_temp = value.(model_copy[counter][:x])
        #println("Obj value: "*string(z_temp)*" x_opt: "*string(x_star_temp[1])*", "*string(x_star_temp[2])*", " *string(x_star_temp[3])*" p: "*string(p_temp))
        # New_Point_List, New_z_list, new_p_list = update_iteration(point_list, z_list, p_list, buffer+i, model_copy[counter], BA, N)
        #println("Iteration "*string(i+buffer))
        #print(model_copy[i])
        if mod(i+1, nrow) == 0 
            unis,uni_z,uni_p, indicators = next_generation(parent_Point_List, child_Point_List, child_z_list, child_p_list, parent_z_list, parent_p_list, indicators)
            comp_point_list, comp_z_list, comp_p_list = add_to_comp(comp_point_list, comp_z_list, comp_p_list, child_Point_List, child_z_list, child_p_list)
            #println("uni is $unis")
            #println("comp is $comp_point_list")
        end
    end
    return comp_point_list, comp_z_list, comp_p_list
end

function is_initialized!(array::AbstractArray)
    nrows, ncol = size(array)
    litmus = fill(-1,3)
    check = findfirst.(.==(vec(litmus)'), eachrow(array))
    check =nothing_to_zero(check)
    rows = Vector{Int64}(ones(nrows))
    checker = all!(rows,check)
    if any(checker.==1)
        for i in eachindex(checker)
            indx = findfirst(==(1), checker)
            deleteat!(array, indx)
        end
    end
end

function nothing_to_zero(x::AbstractArray)
    row,col = size(x)
    x = convert(Array{Union{Int64,Bool,Nothing}},x)
    for i in 1:row
        for j in 1:col
            if isnothing(x[i,j])
                x[i,j] = false
            end
        end
    end
    return x
end

function next_generation(parents::AbstractArray, children::AbstractArray, childz::AbstractVector, childp::AbstractVector,z::AbstractVector, p::AbstractVector, indic::AbstractArray)
    nchil, nvar = size(children)
    is_initialized!(children)
    for i in 1:nchil
        check = findfirst.(.==(children[i,:]'), parents)
        nrow, ncol = size(parents)
        check =nothing_to_zero(check)
        rows = Vector{Int64}(ones(nrow))
        checker = all!(rows,check)
        #println("Checker = $checker")
        if any(checker.==1)
            local j, indx
            indx = findfirst(==(1), checker)
            if indic[indx,1] == 1
                global indic[indx,2] = 1
            else
                global indic[indx,1] = 1
            end
        else
            parents = [parents;children[i,:]']
            indic = [indic;[0 0]]
            z = [z;childz[i]]
            p = [p;childp[i]]
        end
    end
    #println(parents)
    #println(children)
    parents, z, p, count = find_unique(parents, z, p)
    parents = Matrix(transpose(parents))
    #println("Indicator is $indic")
    return parents, z, p, indic
end

function add_to_comp(compiled_points::AbstractArray, compiled_z::AbstractVector, compiled_p::AbstractVector, child_points::AbstractArray, child_z::AbstractArray, child_p::AbstractVector)
    rows, cols = size(child_points)
    for i in 1:rows
        check = findfirst.(.==(child_points[i,:]'), compiled_points)
        pointcount, vars=size(compiled_points)
        check =nothing_to_zero(check)
        count = Vector{Int64}(ones(pointcount))
        checker = all!(count,check)
        if any(checker.==1)
            continue
        else
            compiled_points = [compiled_points;child_points[i,:]']
            compiled_z = [compiled_z;child_z[i]]
            compiled_p = [compiled_p;child_p[i]]
        end
    end
    return compiled_points, compiled_z, compiled_p
end

function BAObj(initial_points::AbstractArray, point_list::AbstractArray, z::AbstractVector, p::AbstractVector, cent::AbstractVector, indic::AbstractArray)
    #println("Indic is: $indic")
    #println("Points are: $point_list")
    x_star = point_list[1,:]
    init_pointnum, varnum = size(initial_points)
    nrow, ncol = size(point_list)
    points_only = initial_points[1:end, 1:ncol]
    local viable_parents = Array{Float64,2}(undef, (0, ncol))
    local viable_indics = Vector{Float64}(undef, 0)
    #global viable_indics = viable_indics'
    tol = 10
    indicator = 0
    for i in 1:nrow
        if indic[i,2] == 0
            viable_parents = [viable_parents;point_list[i,:]']
            viable_indics = [viable_indics;indic[i,1]]
        end
    end

    rows = 1:length(viable_indics)
    if length(viable_indics) == 2
        viable_parents = point_list
        viable_indics = indic
    end
    point_order = shuffle(collect(combinations(rows, 2)))
    init_points_ord = shuffle(collect(combinations(1:init_pointnum, 2)))
    mids = fill(-1.0, (length(point_order), ncol))
    local init_mids = fill(-1.0, (init_pointnum, ncol))
    local init_quarts = fill(-1.0, (2*init_pointnum, ncol))
    objs = fill(-1.0, (length(point_order), ncol))
    endpoints = fill(-1, (length(point_order), 3))
    for i in 1:init_pointnum
        init_mids[i,:] = 0.5*(points_only[init_points_ord[i][1],:] + points_only[init_points_ord[i][2],:])
        init_quarts[2*i-1,:] = 0.5*(points_only[init_points_ord[i][1],:] + init_mids[i,:])
        init_quarts[2*i,:] = 0.5*(points_only[init_points_ord[i][2],:] + init_mids[i,:])
    end
    opts = [init_mids;init_quarts]
    optnum, vars2 = size(opts)
    for i in eachindex(point_order)
        mids[i,:] = 0.5*(viable_parents[point_order[i][1],:] + viable_parents[point_order[i][2],:])
        endpoints[i,:] = [point_order[i][1], point_order[i][2], 0]
        #sel = select_projection(indic[i,1], length(rows), endpoints[i,:], init_pointnum)
        #if init_mids[sel,:] == mids[i,:]
        a = rand(1:optnum)
        if vec(opts[a,:]) in mids[i,:]
            a = mod(a,i) + 1
        end
        objs[i,:] = mids[i,:] - vec(opts[a,:])#cent#points_only[a,:]#
        #else
            #objs[i,:] = mids[i,:] - vec(init_mids[sel,:])
        #end
    end
    b = 2 #float(rand(0:2))
    counti,countj = size(objs)
    for i in 1:counti
        for j in 1:ncol
            if  isapprox(objs[i,j],0) && indicator == 0
                objs[i,j] = b
                indicator = 1
            end
        end
        indicator = 0
    end
    #plt = plot_objs(viable_parents, z,cent, mids, objs)
    #display(plt)
    return objs, viable_parents, endpoints
end

function collect_objs(opts::AbstractArray, mid::AbstractVector)
    nrow, ncol = size(opts)
    objs = fill(-1,(nrow,ncol))
    for i in 1:ncol
        for j in 1:nrow
            objs[j,i] = mid[i] - opts[j,i]
        end
    end
    return objs
end

function find_norm(objs::AbstractArray, side_vec::AbstractVector)
    nrow, ncol = size(objs)
    obj_vec = fill(-1.0, ncol)
    angs = fill(-1.0, nrow)
    local indx = 1
    for i in 1:nrow
        angs[i] = acosd(dot(vec(objs[i,:]), side_vec)/(norm(side_vec)*norm(vec(objs[i,:]))))
        if abs(90-angs[i]) <= abs(90-angs[indx])
            indx = i
        end
    end
    
    println(objs)
    println(angs)
    println(indx)
    obj_vec = objs[indx,:]
    println("Side is: $side_vec")
    println("Obj is : $obj_vec")
    println("Angle is: "*string(angs[indx]))
    return obj_vec
end

function select_projection(indic::Int64, leng::Int64, endpoints::AbstractArray, ninitial_points::Int64)
    A = Vector(1:leng)
    nrow = ninitial_points
    B = Vector(1:nrow)
    deleteat!(A, A .== endpoints[1])
    deleteat!(B, B .== endpoints[1])
    deleteat!(A, A .== endpoints[2])
    deleteat!(B, B .== endpoints[2])
    if indic == 0
        nrow = length(B)
        for j = 2:nrow
            deleteat!(B,2)
        end
    elseif indic == 1
        deleteat!(B,1)
        nrow = length(B)
        for j = 2:nrow
            deleteat!(B,2)
        end
    end
    return B
end

function plot_objs(points::AbstractArray, z::AbstractVector, cent::AbstractVector, mids::AbstractArray, objs::AbstractArray)
    plot([mesh3d(
        x=points[:,1],
        y=points[:,2],
        z=points[:,3],
        colorbar_title="Obj",
        colorscale=[[2, "gold"],
                    [2.2, "magenta"]],
        intensity=z,
        alphahull = 0,
        name="y",
        showscale=true
    ),
    scatter3d(x=points[:,1],
    y=points[:,2],
    z=points[:,3],
    mode="markers",
    marker=attr(
        size = 8,
        color="#000000",
        opacity =0),
        type="scatter3d"
    ),
    scatter3d(
        x=cent[1],
        y=cent[2],
        z=cent[3],
        mode="markers",
        marker=attr(
        size = 8,
        opacity =0),
        type="scatter3d"
    ),
    cone(
    x=mids[:,1],
    y=mids[:,2],
    z=mids[:,3],
    u=objs[:,1],
    v=objs[:,2],
    w=objs[:,3],
    sizemode="absolute",
    sizeref=2,
    anchor="base")
])
end

function timed_capacityminmax(Orig::MGASandbox.MGAInfo, iterations::Int64, slack::Float64, print::Bool)
    solve_time = 0.0
    obj_time = 0.0
    method = 6
    iterations = check_it_num(iterations, length(Orig.nvars))
    

    start_time = time_ns()
    New_Point_List, New_z_list, new_p_list = initialize_lists(Orig.point_list, Orig.z, iterations, Orig.nvars)
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

    for i in 1:iterations
        model_copy = copy(Orig.orig_model)
        set_optimizer(model_copy, CPLEX.Optimizer)
        model_copy = set_slack(model_copy, Orig.z, float(slack))
        prev_avg_dist = avg_dist
        counter += 1

        @objective(model_copy, Min, sum(a[i,j]*all_variables(model_copy)[j] for j in eachindex(Orig.nvars)))
        init_time2 = time_ns()
        optimize!(model_copy)
        solve_time = time_ns() - init_time2
        """
        avg_dist = est_chull_vol(New_Point_List)#evaluate_manhattan_dist(current_point,New_Point_List) #estimate_vol(New_Point_List) #
        if i == 1
            conv_crit = avg_dist/50
        end 
        push!(dists,avg_dist)
        """
        New_Point_List, New_z_list, new_p_list = update_iteration(New_Point_List, New_z_list, new_p_list, i, model_copy, false, false)

    end
    tot_time = time_ns() - start_time
    times = MGASandbox.timing(init_time, obj_time, solve_time, tot_time, 0, counter)

    if print == true
        # print_dists(dists,method, length(Orig.nvars), conv_vec)
        return New_z_list, New_Point_List, new_p_list, times #, a 
    else
        return New_z_list, New_Point_List, new_p_list, times, a, dists, conv_vec
    end
    
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


"""
Post Processing Helper Functions
"""

function find_unique(points::AbstractArray,z::AbstractVector,p::AbstractVector)
    println(points)
    pointst = transpose(points)
    nrow, ncol = size(pointst)
    uniques = fill(-1.0, (nrow, ncol))
    z_uni = Vector{Float64}(undef, ncol)
    p_uni = Vector{Float64}(undef, ncol)
    counter=0
    for i in 1:ncol
        for k in 1:ncol
            if isapprox(pointst[:,i],uniques[:,k],atol=0.001)
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
    #uniquesT = transpose(uniques)
    return uniques, z_uni, p_uni, counter
end

function plot_space(points::AbstractArray, z::AbstractVector)
    plot([mesh3d(
        x=points[1,:],
        y=points[2,:],
        z=points[3,:],
        colorbar_title="Obj",
        colorscale=[[2, "gold"],
                    [2.2, "magenta"]],
        intensity=z,
        alphahull = 0,
        name="y",
        showscale=true
    ),
    scatter3d(x=points[1,:],
    y=points[2,:],
    z=points[3,:],
    mode="markers",
    marker=attr(
        size = 8,
        color="#000000",
        opacity =0),
        type="scatter3d"
    )])
end

function estimate_vol(points::AbstractArray)
    (nrow,ncol) = size(points)
    maxmin = Array{Float64,2}(undef,(2,ncol))
    dists = Vector{Float64}(undef, 0)
    for j in 1:ncol
        maxmin[1,j] = maximum(points[:,j])
        maxmin[2,j] = minimum(points[:,j])
        push!(dists, maxmin[1,j]-maxmin[2,j])
    end
    vol_est = prod(dists)
    return vol_est
end

function est_chull_vol(points::AbstractArray)
	(nit, nvars) = size(points)
	pairs = collect(combinations(1:nvars, 2))
	pairwise_caps = fill(0.0, (nit,2))
	areas = Vector{Float64}(undef, 0)
	tot_area = 0.0
	for i in eachindex(pairs)
		pairwise_caps[:,1] = view(points,:,pairs[i][1])
		pairwise_caps[:,2] = view(points,:,pairs[i][2])
        poly = polyhedron(vrep(pairwise_caps))
        println(poly)
		vol = 0.0
        vol = Polyhedra.volume(poly)
        push!(areas, vol)
	end
    println(areas)
	tot_area = sum(areas[i] for i in eachindex(areas))
    println(tot_area)
	return tot_area
end

function compare_vol(points::AbstractArray)
    est = est_chull_vol(points)
    P = polyhedron(vrep(points))
    act = Polyhedra.volume(P)
    return [est act]

end

function volume_comparison_test(points::AbstractArray)
    (nrow, ncol) = size(points)
    est_df = DataFrame(Estimate = 0.0, Actual = 0.0)
    for i in 6:nrow
        subset = points[1:i,:]
        vols = compare_vol(subset) 
        push!(est_df, vols)
    end
    println(est_df)
    return est_df
end

"""
Main Running Function
"""

function main()
    # Settings
    iterations = 50
    slack =  3.0
    initial_its = 1 # 1 or greater

    #model, nvars = LP_3D()
    #z, x_star = solve_orig_obj2(model)
    model, nvars = LP_N(5)
    z = objective_value(model)
    x_star = value.(model[:x])
    point_list = Array{Float64, 2}(undef, iterations+1, length(nvars))
    for i in eachindex(nvars)
        point_list[1,i] = x_star[i]
    end
    Orig_info = MGAInfo(model, point_list, z, nvars)

    """ Method Choice """
    #z_list, point_list, p_list =  hsj(Orig_info, iterations, float(slack)) # Use if HSJ
    point_list, z_list, p_list = Heuristic_MGA(Orig_info, iterations, float(slack)) # Use if heuristic
    #z_list, point_list, p_list = Distance(Orig_info, iterations, float(slack))
    #point_list, z_list, p_list = MAA(Orig_info, iterations, slack, initial_its) 
    # point_list, z_list, p_list, times = BAMGA(Orig_info, iterations, float(slack), initial_its)
    #z_list, point_list, p_list, times= timed_capacityminmax(Orig_info, iterations, float(slack), true)
    #z_list, point_list, p_list = SPORES(Orig_info, iterations, slack)

    #println(z_list)
    #println(p_list)
    
    uniques, z_uni, p_uni, num_uni = find_unique(point_list, z_list, p_list)
    """
    vol = estimate_vol(uniques)
    println("Volume is: vol")
    plt = plot_space(uniques, z_uni)
    display(plt)
    
    npoints = length(z_list)
    println(npoints)
    """
    uniquesT = transpose(uniques)
    println(uniquesT)
    est_df = volume_comparison_test(uniquesT)
    folder = "C:\\Users\\mike_\\Documents\\ZeroLab\\MGA-Methods"
    output = "Outputs"
    outpath = joinpath(folder, output)
    run = "vol_comp.csv"
    overallpath = joinpath(outpath, run)
    #uniques_df = DataFrame(uniquesT,:auto)
    CSV.write(overallpath, est_df)
end

@time main()
end



