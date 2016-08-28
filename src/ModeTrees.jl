# Data structure

export ModeTree

abstract AbstractModeTree

typealias ModeDict Dict{Tree,Vector{Mode}}
immutable ModeTree <: AbstractModeTree
    tree::Tree
    free_modes::ModeDict
end
immutable SquaredModeTree <: AbstractModeTree
    mtree::ModeTree
end


# Basic functions

export tree, unsquare

tree(t::ModeTree) = t.tree
tree(t::SquaredModeTree) = tree(t.mtree)

Tensors.square(t::ModeTree) = SquaredModeTree(t)
unsquare(t::AbstractModeTree) = t
unsquare(t::SquaredModeTree) = t.mtree

for f in (:getindex, :setindex!) @eval $f(x::ModeTree, args...) = $f(x.free_modes, args...) end
getindex(x::SquaredModeTree, v) = square(x.mtree[v]) 


# Tree construction

export ModeTree!, link!

function ModeTree!(free_modes::Vector{Mode}, children::ModeTree...) 
    r = Tree!([tree(u) for u in children]...)
    D = merge!(ModeDict(r => copy(free_modes)), [u.free_modes for u in children]...)
    return ModeTree(r,D)
end
function ModeTree!(free_modes::Vector{Mode})
    r = Tree()
    D = ModeDict(r => free_modes)
    return ModeTree(r,D)
end
ModeTree(free_modes::Vector{Mode}, children::ModeTree...) = ModeTree!(free_modes, deepcopy(children)...)
ModeTree(free_modes::Vector{Mode}) = ModeTree!(free_modes)

function link!(v::ModeTree, u::ModeTree)
    link!(tree(v), tree(u))
    merge!(v.free_modes, u.free_modes)
    return v
end


# Common tree structures

export lintree

function lintree{T <: AbstractVector{Mode}}(D::AbstractVector{T})
    mtree = ModeTree(D[1])
    for i = 2:length(D)
        mtree = ModeTree!(D[i], mtree)
    end
    return mtree
end
lintree(D::AbstractVector{Mode}) = lintree([[k] for k in D])


# Other

function Tensors.mode(t::ModeTree)
    modes = Mode[]
    for D in values(t.free_modes) append!(modes, D) end
    return modes
end
Tensors.mode(t::SquaredModeTree) = square(mode(t.mtree))
