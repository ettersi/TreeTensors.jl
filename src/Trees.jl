export Tree, PairSet, Tree!, link!,
    neighbors, children, neighbor_edges, child_edges,
    edges, vertices, root_to_leaves, leaves_to_root, both_ways


# Tree data structure

type Tree
    neighbors::Vector{Tree}
end
Tree() = Tree([])

show(io::IO, v::Tree) = print(io, "Vertex ", UInt(pointer_from_objref(v)))


# Pair sets (used for edges)

immutable PairSet{T} t::Tuple{T,T} end
PairSet{T}(t1::T, t2::T) = PairSet((t1,t2))
getindex(s::PairSet, i) = s.t[i]
=={T}(s::PairSet{T}, t::PairSet{T}) = (s[1] == t[1] && s[2] == t[2]) || (s[1] == t[2] && s[2] == t[1])
hash(s::PairSet, h::UInt) = hash(s[1],h) $ hash(s[2],h)
show(io::IO, s::PairSet) = print(io, "PairSet{",eltype(s),"}(",s[1],",",s[2],")")

# Allows to write (a,b) = PairSet(a,b)
start(s::PairSet) = start(s.t)
next(s::PairSet, state) = next(s.t, state)
done(s::PairSet, state) = done(s.t, state)
eltype{T}(s::PairSet{T}) = T
length(s::PairSet) = 2


# Tree construction

function Tree!(neighbors::Tree...)
    v = Tree([neighbors...])
    for u in neighbors
        push!(u.neighbors,v)
    end
    return v
end
Tree(neighbors::Tree...) = Tree!(deepcopy(neighbors)...)

function link!(v::Tree,u::Tree) 
    push!(v.neighbors,u)
    push!(u.neighbors,v)
    return v
end


# Property functions

neighbors(v::Tree) = v.neighbors
children(v::Tree, p::Tree) = filter(u -> u != p, neighbors(v))
neighbor_edges(v::Tree) = [PairSet(u,v) for u in neighbors(v)]
child_edges(v::Tree, p::Tree) = [PairSet(u,v) for u in children(v,p)]


# Tree traversal

abstract TraversalOrder
immutable RootToLeaves <: TraversalOrder end
const root_to_leaves = RootToLeaves()
immutable LeavesToRoot <: TraversalOrder end
const leaves_to_root = LeavesToRoot()
immutable BothWays <: TraversalOrder end
const both_ways = BothWays()

function edges_rec(v,p, order::TraversalOrder)
    for u in children(v,p) 
        if isa(order, RootToLeaves) produce(PairSet(u,v)); end
        if isa(order, BothWays) produce(PairSet(v,u)); end
        edges_rec(u,v, order)
        if isa(order, LeavesToRoot) produce(PairSet(u,v)); end
        if isa(order, BothWays) produce(PairSet(u,v)); end
    end
end
edges_with_root(v::Tree, p::Tree, order) = @task begin
    if isa(order, RootToLeaves) produce(PairSet(v,p)) end
    if isa(order, BothWays) produce(PairSet(p,v)); end
    edges_rec(v,p, order)
    if isa(order, LeavesToRoot) produce(PairSet(v,p)) end
    if isa(order, BothWays) produce(PairSet(v,p)); end
end
edges(v::Tree, p::Tree, order) = @task edges_rec(v,p, order)

function vertices_rec(v,p, order::TraversalOrder)
    if isa(order, RootToLeaves) produce(v); end
    if isa(order, BothWays) produce(v); end
    for u in children(v,p) 
        vertices_rec(u,v, order)
    end
    if isa(order, LeavesToRoot) produce(v); end
    if isa(order, BothWays) produce(v); end
end
vertices(v::Tree, p::Tree, order) = @task vertices_rec(v,p, order)

for f in (:edges, :edges_with_root, :vertices)
    @eval begin
        $f(v::Tree, order) = $f(v,v,order)
        $f(x, order) = $f(tree(x),order)

        $f(v::Tree, p::Tree) = $f(v,p,root_to_leaves)
        $f(v::Tree) = $f(v,v)
        $f(x) = $f(tree(x))
    end
end

