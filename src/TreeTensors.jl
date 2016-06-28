module TreeTensors

export TreeTensor, scalartype, modetree, tree, 
    decompose, contract!, contract,
    orthogonalize, orthogonalize!, truncate, truncate!, norm!

importall Base
using Tensors
include("Trees.jl")
include("ModeTrees.jl")


# Tree tensor networks

typealias TensorDict{T} Dict{Tree,Tensor{T}}
immutable TreeTensor{T}
    mtree::AbstractModeTree
    tensors::TensorDict{T}
end
call{T}(::Type{TreeTensor{T}}, mtree) = TreeTensor(mtree, TensorDict{T}())


# Basic functions

Tensors.scalartype{T}(::Type{TreeTensor{T}}) = T
modetree(x::TreeTensor) = x.mtree
tree(x::TreeTensor) = tree(modetree(x))
copy(x::TreeTensor) = TreeTensor(x.mtree, TensorDict{scalartype(x)}(x.tensors))


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
for f in (:ones, :rand)
    @eval $f(mtree::AbstractModeTree, x...) = $f(Float64, mtree, x...)
end
for f in (:ones, :eye)
    @eval $f(x::TreeTensor) = $f(scalartype(x), modetree(x))
end


# Conversion to and from full

function decompose(x::Tensor, mtree, rank)
    y = TreeTensor{scalartype(x)}(mtree)
    for (v,p) in edges(y, leaves_to_root)
        b,s,y[v] = svd(x, [child_edges(v,p);mlabel(mtree[v])], PairSet(v,p), rank)
        x = scale!(b,s)
    end
    y[:root] = x
    return y
end

function contract!(x::TreeTensor)
    for (v,p) in edges(x, leaves_to_root)
        x[p] *= x[v]
    end
    return x[:root]
end
contract(x::TreeTensor) = contract!(copy(x))


# Arithmetic

function +{T}(x::TreeTensor{T}, y::TreeTensor{T})
    @assert modetree(x) == modetree(y) "x and y must have the same mode tree"
    mtree = modetree(x)
    return TreeTensor(
        mtree, 
        (Tree=>Tensor{T})[
            v => padcat(x[v],y[v], neighbor_edges(v)) 
            for v in vertices(mtree, root_to_leaves)
        ]
    )
end

function *{T}(x::TreeTensor{T}, y::TreeTensor{T})
    @assert unsquare(modetree(x)) == unsquare(modetree(y)) "x and y must have the same mode tree"
    mtree = modetree(x)
    return TreeTensor(
        mtree, 
        (Tree=>Tensor{T})[
            v => mergem!(tag(x[v], 1,neighbor_edges(v))*tag(y[v], 2,neighbor_edges(v)), [[tag(1,e),tag(2,e)] => e for e in neighbor_edges(v)])
            for v in vertices(mtree, root_to_leaves)
        ]
    )
end

*(a::Number, x::TreeTensor) = (y = copy(x); y[:root] *= a; return y)
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
                (Tree=>Tensor{T})[
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

function orthogonalize!(x::TreeTensor)
    for (v,p) in edges(x, leaves_to_root)
        x[v],c = qr(x[v], PairSet(v,p))
        x[p] = c*x[p]
    end
    return x
end
orthogonalize(x::TreeTensor) = orthogonalize!(copy(x))

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


# Norm and dot

norm!(x::TreeTensor) = norm(orthogonalize!(x)[:root])
norm( x::TreeTensor) = norm(orthogonalize( x)[:root])

function dot(x::TreeTensor, y::TreeTensor)
    x = TreeTensor(
        modetree(x), 
        (Tree=>Tensor{scalartype(x)})[
            v => tag!(conj(x[v]), :left, neighbor_edges(v)) 
            for v in vertices(x)
        ]
    )
    for (v,p) in edges(x, leaves_to_root)
        x[p] *= x[v]*tag(y[v], :right, neighbor_edges(v))
    end
    return scalar(x[:root]*tag(y[:root], :right, neighbor_edges(tree(y))))
end

end # module
