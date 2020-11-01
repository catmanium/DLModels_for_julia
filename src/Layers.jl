module Layers

#==================
命名規則
・活性化関数：Sigmoid
・損失関数：Sigmoid_with_CrossEntopy
==================#

#params
abstract type Params end
mutable struct Sigmoid_params <: Params
    out::Array
end
mutable struct Affine_params <: Params
    W::Array{Float64,2}
    b::Array{Float64,1}
    in
    dW::Array{Float64,2}
end
mutable struct Sig_and_loss_params <: Params
    s::Array #スコア
    t::Array
end


#レイヤコンストラクタ
function Sigmoid()
    out = []
    return Sigmoid_params(out)
end
function Affine(W,b)
    in=[]
    dW = zeros(size(W,1),size(W,2))
    return Affine_params(W,b,in,dW)            
end
function Sigmoid_with_CrossEntropy()
    s = []
    t = []
    return Sig_and_loss_params(s,t)
end


#forward
function forward(layer::Affine_params,in)
    layer.in = in #逆伝播に使う
    return in * layer.W .+ layer.b'
end
function forward(layer::Sigmoid_params,in)
    out = 1.0 ./ (1.0 .+ exp.(-in)) #逆伝播に使う
    layer.out = out
    return out
end
function forward(layer::Sig_and_loss_params,in)
    s = 1.0 ./ (1.0 .+ exp.(-in))
    layer.s = s
    if length(layer.t) == 0
        #ただの推論
        return s
    end
    l = -layer.t .* log.(s) - (1 .-layer.t) .* log.(1 .-s)
    return l
end

#backward
function backward(layer::Affine_params,din)
    layer.dW = (layer.in)' * din
    return din * (layer.W)'
end
function backward(layer::Sig_and_loss_params,din)
    return layer.s - layer.t
end
function backward(layer::Sigmoid_params,din)
    return din .* layer.out .* (1 .- layer.out)
end
    

end
