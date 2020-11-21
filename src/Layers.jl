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
    v #momentam
    s #RMSProp
end
mutable struct ReLU_params <: Params
    mask_id
end
mutable struct Tanh_params <: Params
    out::Array
end
mutable struct Sig_and_loss_params <: Params
    s::Array #スコア
    t::Array
end
mutable struct mse_params <: Params
    s::Array
    t::Array
end
mutable struct mae_params <: Params
    s::Array
    t::Array
end


#レイヤコンストラクタ
function Sigmoid()
    out = []
    return Sigmoid_params(out)
end
function Affine(W::Array{Float64,2},b::Array{Float64,1})
    in=[]
    v = zeros(size(W))
    s = zeros(size(W))
    dW = zeros(size(W))
    return Affine_params(W,b,in,dW,v,s)            
end
function ReLU()
    mask_id = []
    return ReLU_params(mask_id)
end
function Tanh()
    out = []
    return Tanh_params(out)
end
function Sigmoid_with_CrossEntropy()
    s = []
    t = []
    return Sig_and_loss_params(s,t)
end
function Mean_Squared_Error()
    s = []
    t = []
    return mse_params(s,t)
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
function forward(layer::ReLU_params,in)
    id = findall(in.<=0) #0にするIDを抽出
    layer.mask_id = id
    copy_in = copy(in)
    copy_in[id[1:end]] .= 0

    return copy_in
end
function forward(layer::Tanh_params,in)
    out = tanh.(in)
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
function forward(layer::mse_params,in)
    layer.s = in
    if length(layer.t) == 0
        #ただの推論
        return layer.s
    end
    l = ((layer.t-in).^2)/2

    return l
end

#backward
function backward(layer::Affine_params,din)
    layer.dW = (layer.in)' * din
    return din * (layer.W)'
end
function backward(layer::ReLU_params,din)
    din[layer.mask_id[1:end]] .= 0

    return din
end
function backward(layer::Sigmoid_params,din)
    return din .* layer.out .* (1 .- layer.out)
end
function backward(layer::Tanh_params,din)
    return din .* (1 .- layer.out.^2)
end
function backward(layer::Sig_and_loss_params,din)
    return layer.s - layer.t
end
function backward(layer::mse_params,din)
    return (layer.s - layer.t)
end
    

end
