module TreeTensors

importall Base
using Tensors
include("Trees.jl")
include("ModeTrees.jl")


# Tree tensor networks

export TreeTensor
typealias TensorDict{T} Dict{Tree,Tensor{T}}
immutable TreeTensor{T}
    mtree::AbstractModeTree
    tensors::TensorDict{T}
end
call{T}(::Type{TreeTensor{T}}, mtree) = TreeTensor(mtree, TensorDict{T}())


# Basic functions

export modetree, tree
Tensors.scalartype{T}(::Type{TreeTensor{T}}) = T
modetree(x::TreeTensor) = x.mtree
tree(x::TreeTensor) = tree(modetree(x))
copy(x::TreeTensor) = TreeTensor(x.mtree, TensorDict{scalartype(x)}(x.tensors))
convert{T}(::Type{TreeTensor{T}}, x::TreeTensor) = TreeTensor(x.mtree, convert(TensorDict{T},x.tensors))
copyconvert{T}(::Type{TreeTensor{T}}, x::TreeTensor{T}) = copy(x)
copyconvert{T}(::Type{TreeTensor{T}}, x::TreeTensor) = convert(TreeTensor{T}, x)


# Indexing and iteration

for f in (
    :getindex, :setindex!, 
    :start, :next, :done, 
    :eltype, :length, 
    :keys, :values, :keytype, :valtype
) 
    @eval $f(x::TreeTensor, args...) = $f(x.tensors, args...) 
end
function index(x::TreeTensor, s::Symbol)
    if s == :root return tree(x)
    else throw(KeyError(s)) end
end
getindex(x::TreeTensor, s::Symbol) = x[index(x,s)]
setindex!(x::TreeTensor, xv, s::Symbol) = x[index(x,s)] = xv


# Initialization

ones{T}(::Type{T}, mtree::AbstractModeTree) = TreeTensor(mtree, (Tree=>Tensor{T})[
    v => ones(T, [[Mode(e,1) for e in neighbor_edges(v)]; mtree[v]])
    for v in vertices(mtree, root_to_leaves)
])
eye{T}(::Type{T}, mtree::AbstractModeTree) = TreeTensor(square(mtree), (Tree=>Tensor{T})[
    v => begin
        t = eye(T, mtree[v]); 
        t.modes = [t.modes; [Mode(e,1) for e in neighbor_edges(v)]]; 
        t
    end
    for v in vertices(mtree, root_to_leaves)
])
function rand{T}(::Type{T}, mtree::AbstractModeTree, r)
    evaluate(r::Int,e) = r
    evaluate(r::Dict,e) = r[e]
    return TreeTensor(mtree, (Tree=>Tensor{T})[
        v => rand(T, [[Mode(e,evaluate(r,e)) for e in neighbor_edges(v)]; mtree[v]])
        for v in vertices(mtree, root_to_leaves)
    ])
end
for f in (:ones, :rand, :eye)
    @eval $f(mtree::AbstractModeTree, args...) = $f(Float64, mtree, args...)
end
for f in (:ones, :rand, :eye)
    @eval $f(x::TreeTensor, args...) = $f(scalartype(x), modetree(x), args...)
end


# Conversion to and from full

export decompose
function decompose(x::Tensor, mtree, rank)
    y = TreeTensor{scalartype(x)}(mtree)
    for (v,p) in edges(y, leaves_to_root)
        b,s,y[v] = svd(x, [child_edges(v,p);mlabel(mtree[v])], PairSet(v,p), rank)
        x = scale!(b,s)
    end
    y[:root] = x
    return y
end

export contract!, contract
function contract!(x::TreeTensor)
    for (v,p) in edges(x, leaves_to_root)
        x[p] *= x[v]
    end
    return x[:root]
end
contract(x::TreeTensor) = contract!(copy(x))


# Arithmetic

function +(x::TreeTensor, y::TreeTensor)
    @assert modetree(x) == modetree(y) "x and y must have the same mode tree"
    mtree = modetree(x)
    return TreeTensor(
        mtree, 
        (Tree=>Tensor{promote_type(scalartype(x), scalartype(y))})[
            v => padcat(x[v],y[v], neighbor_edges(v)) 
            for v in vertices(mtree, root_to_leaves)
        ]
    )
end

function *(x::TreeTensor, y::TreeTensor)
    @assert unsquare(modetree(x)) == unsquare(modetree(y)) "x and y must have the same mode tree"
    mtree = modetree(x)
    return TreeTensor(
        mtree, 
        (Tree=>Tensor{promote_type(scalartype(x),scalartype(y))})[
            v => mergem!(tag(x[v], 1,neighbor_edges(v))*tag(y[v], 2,neighbor_edges(v)), [[tag(1,e),tag(2,e)] => e for e in neighbor_edges(v)])
            for v in vertices(mtree, root_to_leaves)
        ]
    )
end

*(a::Number, x::TreeTensor) = (y = copyconvert(TreeTensor{promote_type(typeof(a),scalartype(x))}, x); y[:root] *= a; return y)
*(x::TreeTensor, a::Number) = a*x
scale!(a::Number, x::TreeTensor) = (x[:root] *= a; return x)
scale!(x::TreeTensor, a::Number) = scale!(a,x)
-(x::TreeTensor, y::TreeTensor) = x + (-1)*y

conj(x::TreeTensor) = TreeTensor(x.mtree, (Tree=>Tensor{scalartype(x)})[v => conj(xv) for (v,xv) in x])


# Transposition and conjugation

for f in (:conj,:transpose,:ctranspose)
    f! = symbol(string(f)*"!")
    @eval begin
        function Base.$f(t::TreeTensor)
            return TreeTensor(
                modetree(t),
                (Tree=>Tensor{scalartype(t)})[
                    v => $f(tv)
                    for (v,tv) in t
                ]
            )
        end
        function Base.$f!(t::TreeTensor)
            for v in keys(t)
                t[v] = $f!(t[v])
            end
            return t
        end
    end
end


# Orthogonalisation and truncation

export orthogonalize!, orthogonalize
function orthogonalize!(x::TreeTensor)
    for (v,p) in edges(x, leaves_to_root)
        x[v],c = qr(x[v], PairSet(v,p))
        x[p] = c*x[p]
    end
    return x
end
orthogonalize(x::TreeTensor) = orthogonalize!(copy(x))

export truncate!, truncate
function truncate!(x::TreeTensor, rank)
    orthogonalize!(x)
    s = Dict{PairSet{Tree}, Tensor{real(scalartype(x))}}()
    for (v,p) in edges_with_root(x, root_to_leaves)
        for u in children(v,p)
            e = PairSet(u,v)
            b,s[e],d = svd(x[v], e, maxrank())
            x[u] = scale!(s[e],d)*x[u]
            x[v] = scale!(b,s[e])
        end
        x[v] = resize(x[v], [e => rank(s[e].data) for e in neighbor_edges(v)])
        scale!(x[v], [resize(1./s[e], Dict(e => msize(x[v],e))) for e in child_edges(v,p)]...)
    end
    return x
end
truncate(x::TreeTensor, rank) = truncate!(copy(x), rank)


# Contracted subtrees

function contracted_subtree(v::Tree,p::Tree, c::TreeTensor, args::TreeTensor...)
    cv = retag!(tag!(conj(args[1][v]), 1, neighbor_edges(v)), :C => :_, :R => :C)
    for u in children(v,p) cv *= c[u] end
    for i = 2:length(args)-1 cv *= tag(args[i][v], i, neighbor_edges(v)) end
    cv *= retag!(tag(args[end][v], length(args), neighbor_edges(v)), :C => :_)
    return cv
end

function contracted_subtrees(args::TreeTensor...)
    c = TreeTensor(
        modetree(args[1]), 
        TensorDict{scalartype(args[1])}()
    )
    for (v,p) in edges(c, leaves_to_root)
        c[v] = contracted_subtree(v,p,c,args...)
    end
    return c
end


# Norm and dot

export norm!
norm!(x::TreeTensor) = norm(orthogonalize!(x)[:root])
norm( x::TreeTensor) = norm(orthogonalize( x)[:root])

function dot(x::TreeTensor, y::TreeTensor)
    return scalar(
        contracted_subtree(
           tree(x),tree(x), 
           contracted_subtrees(x,y),
           x,y
        )
    )
end


# Local solvers

function localmatrix(A, v, xAx)
    lA = tag(A[v], 2, neighbor_edges(v))
    for u in neighbors(v)
        lA *= xAx[u]
    end
    retag!(lA, 1 => :R, 3 => :C)
    return lA
end

function localrhs(b, v, xb)
    lb = tag(b[v], 2, neighbor_edges(v))
    for u in neighbors(v)
        lb *= xb[u]
    end
    untag!(lb, 1)
    return lb
end

function localmatvecfunc(A, v, xAx)
    Av = tag(A[v],2,neighbor_edges(v))
    c = [retag(xAx[u], 3 => :C, 1 => :R) for u in neighbors(v)]
    if length(c) == 3
        return xv -> begin
            (c[3]*((c[1]*Av)*(c[2]*xv)))
        end
    elseif length(c) == 2
        return xv -> begin
            (c[2]*(Av*(c[1]*xv)))
        end
    elseif length(c) == 1
        return xv -> begin
            (c[1]*Av*xv)
        end
    else
        error("Local matvec structure not implemented!")
    end
end

function localproblem(x,A,b, v, xAx,xb)
    modes = x[v].modes
    matvec = localmatvecfunc(A, v, xAx)
    return x[v][mlabel(modes)], 
           xv -> (xv = Tensor(modes,xv); matvec(xv)[mlabel(modes)]), 
           localrhs(b,v,xb)[mlabel(modes)]
end


# Local solvers

using IterativeSolvers

abstract LocalSolver

export DenseSolver
immutable DenseSolver <: LocalSolver end
function localsolve!(x,A,b, solver::DenseSolver, v, xAx, xb)
    x[v] = localmatrix(A,v,xAx)\localrhs(b,v,xb)
end

export GMRES
immutable GMRES <: LocalSolver
    tol::Float64
    maxiter::Int
    restart::Int
    warn::Bool
end
GMRES(; tol = sqrt(eps()), maxiter = 1, restart = 20, warn = false) = GMRES(tol, maxiter, restart, warn)
function localsolve!(x,A,b, solver::GMRES, v, xAx,xb)
    _,hist = gmres!(
        localproblem(x,A,b, v, xAx,xb)...; 
        tol = solver.tol,
        maxiter = solver.maxiter, 
        restart = solver.restart
    )
    if !hist.isconverged && solver.warn
        warn(
            "Local GMRES iteration did not convergence.\n"*
            "  LSE size:  "*string(length(x[v]))*"\n"*
            "  Threshold: "*string(hist.threshold)*"\n"*
            "  Residuals: "*string(hist.residuals)
        )
    end
end

export CG
immutable CG <: LocalSolver
    tol::Float64
    maxiter::Int
    warn::Bool
end
CG(; tol = sqrt(eps()), maxiter = 20, warn = false) = CG(tol, maxiter, warn)
function localsolve!(x,A,b, solver::CG, v, xAx,xb)
    _,hist = cg!(
        localmatvec(x,A,b, v, xAx,xb)...;
        tol = solver.tol,
        maxiter = solver.maxiter
    )
    if !hist.isconverged && solver.warn
        warn(
            "Local CG iteration did not convergence.\n"*
            "  LSE size:  "*string(length(x[v]))*"\n"*
            "  Threshold: "*string(hist.threshold)*"\n"*
            "  Residuals: "*string(hist.residuals)
        )
    end
end


# ALS linear solver

export als_solve!
function als_solve!(
    x,A,b, solver = GMRES(); 
    maxiter::Int = 20, tol = sqrt(eps(real(scalartype(x))))
)
    updatenorms = zeros(real(scalartype(x)), maxiter)

    orthogonalize!(x)
    xAx = contracted_subtrees(x,A,x)
    xb = contracted_subtrees(x,b)
    for i = 1:maxiter
        for (u,v) in edges(x, both_ways)
            # Solve
            xu = copy(x[u])
            localsolve!(x,A,b, solver, u, xAx,xb)
            updatenorms[i] = max(updatenorms[i], norm(xu - x[u]))

            # Orthogonalize
            x[u],r = qr(x[u],PairSet(u,v))
            x[v] = r*x[v]

            # Compute contracted subtrees
            xAx[u] = contracted_subtree(u,v, xAx, x,A,x)
            xb[u] = contracted_subtree(u,v, xb, x,b)
        end

        # Convergence check
        if updatenorms[i] < tol*norm(x[:root])
            return x, ConvergenceHistory(true, tol*norm(x[:root]), i, updatenorms[1:i])
        end
    end
    return x, ConvergenceHistory(false, tol*norm(x[:root]), maxiter, updatenorms)
end


# ALS sum

export als_sum!
function als_sum!{T}(
    y,x::AbstractVector{TreeTensor{T}};
    maxiter::Int = 20, tol = sqrt(eps(real(T)))
)
    updatenorms = zeros(real(T), maxiter)

    orthogonalize!(y)
    yx = [contracted_subtrees(y,x) for x in x]
    for i = 1:maxiter
        for (u,v) in edges(y, both_ways)
            # Project
            yunew = sum([localrhs(x, u, yx) for (x,yx) in zip(x,yx)])
            updatenorms[i] = max(updatenorms[i], norm(yunew - y[u]))
            y[u] = yunew

            # Orthogonalize
            y[u],r = qr(y[u],PairSet(u,v))
            y[v] = r*y[v]

            # Compute contracted subtrees
            for (x,yx) in zip(x,yx)
                local x,yx
                yx[u] = contracted_subtree(u,v, yx, y,x)
            end
        end

        # Convergence check
        if updatenorms[i] < tol*norm(y[:root])
            return y, ConvergenceHistory(true, tol*norm(y[:root]), i, updatenorms[1:i])
        end
    end
    return y, ConvergenceHistory(false, tol*norm(y[:root]), maxiter, updatenorms)
end


# ALS AXPY

export als_axpy!
function als_axpy!(
    z,A,x,y;
    maxiter::Int = 20, tol = sqrt(eps(real(scalartype(x))))
)
    updatenorms = zeros(real(scalartype(x)), maxiter)

    orthogonalize!(z)
    zAx = contracted_subtrees(z,A,x)
    zy = contracted_subtrees(z,y)
    for i = 1:maxiter
        for (u,v) in edges(z, both_ways)
            # Project
            zunew = localmatvecfunc(A, u, zAx)(x[u]) + localrhs(y, u, zy)
            updatenorms[i] = max(updatenorms[i], norm(zunew - z[u]))
            z[u] = zunew

            # Orthogonalize
            z[u],r = qr(z[u],PairSet(u,v))
            z[v] = r*z[v]

            # Compute contracted subtrees
            zAx[u] = contracted_subtree(u,v, zAx, z,A,x)
            zy[u] = contracted_subtree(u,v, zy, z,y)
        end

        # Convergence check
        if updatenorms[i] < tol*norm(z[:root])
            return z, ConvergenceHistory(true, tol*norm(z[:root]), i, updatenorms[1:i])
        end
    end
    return z, ConvergenceHistory(false, tol*norm(z[:root]), maxiter, updatenorms)
end


## ALSSD linear solver
#
#function alssd_solve!(
#    x,A,b, solver = GMRES(); 
#    residualrank = 4, maxiter::Int = 20, tol = sqrt(eps(real(scalartype(x))))
#)
#    residualnorms = zeros(real(scalartype(x)), maxiter)
#    tol *= norm(b)
#
#    z = rand(x, residualrank)
#    for i = 1:maxiter
#        als_axpy!(z, -A,x,b; maxiter = 1)
#
#        residualnorms[i] = norm!(z)
#        if residualnorms[i] < tol
#            return x,ConvergenceHistory(true, tol, i, residualnorms[1:i])
#        end
#
#        x += z
#        als_solve!(x,A,b, solver; maxiter = 1)
#        truncate!(x, adaptive(tol))
#    end
#    return x
#end


## ALS operator inversion
#
#function als_inv!(
#    X,A, solver = GMRES(); 
#    maxiter::Int = 20, tol = sqrt(eps(real(scalartype(X))))
#)
#    updatenorms = zeros(real(scalartype(X)), maxiter)
#
#    orthogonalize!(X)
#    XAX = contracted_subtrees(X,A,X)
#    XAXt = contracted_subtrees(X,A,X)
#    xb = contracted_subtrees(x,b)
#    for i = 1:maxiter
#        for (u,v) in edges(x, both_ways)
#            # Solve
#            xu = copy(x[u])
#            localsolve!(x,A,b, solver, u, xAx,xb)
#            updatenorms[i] = max(updatenorms[i], norm(xu - x[u]))
#
#            # Orthogonalize
#            x[u],r = qr(x[u],PairSet(u,v))
#            x[v] = r*x[v]
#
#            # Compute contracted subtrees
#            xAx[u] = contracted_subtree(u,v, xAx, x,A,x)
#            xb[u] = contracted_subtree(u,v, xb, x,b)
#        end
#
#        # Convergence check
#        if updatenorms[i] < tol*norm(x[:root])
#            return x, ConvergenceHistory(true, tol*norm(x[:root]), i, updatenorms[1:i])
#        end
#    end
#    return x, ConvergenceHistory(false, tol*norm(x[:root]), maxiter, updatenorms)
#end

end # module
