module Layers

#==================
命名規則
・活性化関数：Sigmoid
・損失関数：Sigmoid_with_CrossEntopy

共通メンバ
・params : 重み，バイアス
・grads : 勾配

共通メソッド
・forward()
・backward()

損失レイヤ
    ・メンバ
        ・t : 教師データ
        ・s : スコア
    ・メソッド
        ・forward() : t=[]ならスコア，あるなら損失を返す
==================#

#params
abstract type Params end
mutable struct Sigmoid_params <: Params
    params
    grads
    out::Array
end
mutable struct Affine_params <: Params
    params #W,b
    grads #dW,db
    in
end
mutable struct ReLU_params <: Params
    params
    grads
    mask_id
end
mutable struct Tanh_params <: Params
    params
    grads
    out::Array
end
mutable struct Sig_and_loss_params <: Params
    params
    grads
    s::Array #スコア
    t::Array
end
mutable struct mse_params <: Params
    params
    grads
    s::Array
    t::Array
end
mutable struct RNN_params <: Params
    Wx::Array{Float64,2}
    Wh::Array{Float64,2}
    b::Array{Float64,1}
    dWx::Array{Float64,2}
    dWh::Array{Float64,2}
    cache
end
mutable struct TimeRNN_params <: Params
    Wx::Array{Float64,2}
    Wh::Array{Float64,2}
    b::Array{Float64,1}
    dWx::Array{Float64,2}
    dWh::Array{Float64,2}
    #複数のRNNレイヤを管理
    layers
    h
    dh
    stateful
end
mutable struct TimeAffine_params <: Params
    W::Array{Float64,2}
    b::Array{Float64,1}
    dW::Array{Float64,2}
    layers
end
mutable struct TimeMse_params <: Params
    layers
end


#レイヤコンストラクタ
function Sigmoid()
    out = []
    params = []
    grads = []
    return Sigmoid_params(params,grads,out)
end
function Affine(params,grads)
    in=[] 
    return Affine_params(params,grads,in)            
end
function ReLU()
    params = []
    grads = []
    mask_id = []
    return ReLU_params(params,grads,mask_id)
end
function Tanh()
    params = []
    grads = []
    out = []
    return Tanh_params(params,grads,out)
end
function Sigmoid_with_CrossEntropy()
    params = []
    grads = []
    s = []
    t = []
    return Sig_and_loss_params(params,grads,s,t)
end
function Mean_Squared_Error()
    params = []
    grads = []
    s = []
    t = []
    return mse_params(params,grads,s,t)
end
function RNN(Wx::Array{Float64,2},Wh::Array{Float64,2},b::Array{Float64,1})
    dWx = zeros(size(Wx))
    dWh = zeros(size(Wh))
    cache = []
    return RNN_params(Wx,Wh,b,dWx,dWh,cache)
end
function TimeRNN(Wx::Array{Float64,2},Wh::Array{Float64,2},b::Array{Float64,1},stateful::Bool)
    dWx = zeros(size(Wx))
    dWh = zeros(size(Wh))
    layers = nothing
    dh = nothing
    h = nothing
    return TimeRNN_params(Wx,Wh,b,dWx,dWh,layers,dh,h,stateful)
end
function TimeAffine(W::Array{Float64,2},b::Array{Float64,1})
    dW = zeros(size(W))
    layers = []
    return TimeAffine_params(W,b,dW,layers)
end
function TimeMse()
    return TimeMse_params([])
end


#forward
function forward(layer::Affine_params,in)
    layer.in = in #逆伝播に使う
    return in * layer.params[1] .+ layer.params[2]'
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
function forward(layer::RNN_params,x,h_prev)
    t = h_prev*layer.Wh + x*layer.Wx .+ layer.b'
    h_next = tanh.(t)

    layer.cache = [x,h_prev,h_next]

    return h_next
end
function forward(layer::TimeRNN_params,xs)
    Wx = layer.Wx
    Wh = layer.Wh
    b = layer.b
    #バッチサイズ，ブロック数，入力データ数
    N, T, D = size(xs)
    D, H = size(Wx)

    layer.layers = []
    hs = zeros(Float64,(N,T,H))

    if !layer.stateful || layer.h === nothing
        layer.h = zeros(Float64,(N,H))
    end

    for t in 1:T
        rnn_layer = RNN(layer.Wx,layer.Wh,layer.b)
        layer.h = forward(rnn_layer,xs[:,t,:],layer.h)
        hs[:,t,:] = layer.h
        push!(layer.layers,rnn_layer)
    end

    return hs
end
function forward(layer::TimeAffine_params,hs)
    N,T,H = size(hs)
    H,O = size(layer.W)
    ys = zeros(Float64,(N,T,O))

    for t in 1:T
        h = hs[:,t,:]
        affine_layer = Affine(layer.W,layer.b)
        y = forward(affine_layer,h)
        ys[:,t,:] = y
        push!(layer.layers,affine_layer)
    end

    return ys
end
function forward(layer::TimeMse_params,ys,t_data)
    N,T,O = size(ys)   
    loss = zeros(Float64,size(ys))
    
    for t in 1:T
        mse_layer = Mean_Squared_Error()
        mse_layer.t = t_data[:,t,:]
        y = ys[:,t,:]
        loss[:,t,:] = forward(mse_layer,y)
        push!(layer.layers,mse_layer)
    end

    return loss
end

#backward
function backward(layer::Affine_params,din)
    layer.grads[1] .= (layer.in)' * din
    return din * (layer.params[1])'
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
function backward(layer::RNN_params,dh_next)
    Wx, Wh, b = layer.Wx, layer.Wh, layer.b
    x, h_prev, h_next = layer.cache

    dt = dh_next .* (1 .- h_next .^2)
    dWh = h_prev' * dt
    dh_prev = dt * Wh'
    dWx = x' * dt
    dx = dt * Wx'

    layer.dWx = dWx
    layer.dWh = dWh

    return dx,dh_prev
end
function backward(layer::TimeRNN_params,dhs)
    Wx = layer.Wx
    Wh = layer.Wh
    b = layer.b
    N, T, H = size(dhs)
    D, H = size(Wx)

    dxs = zeros(Float64,(N,T,D))
    dh = zeros(Float64,size(dhs[:,1,:]))

    for t in T:-1:1
        rnn_layer = layer.layers[t]
        dx, dh = backward(rnn_layer,dhs[:,t,:]+dh)
        dxs[:,t,:] = dx

        #勾配合算
        layer.dWx .+= rnn_layer.dWx
        layer.dWh .+= rnn_layer.dWh
    end

    return dxs
    
end
function backward(layer::TimeAffine_params,dys)
    N, T, O = size(dys)
    H, O = size(layer.layers[1].W)
    dhs = zeros(Float64,(N,T,H))

    for t in 1:T
        dh = zeros(Float64,(N,H))
        dh = backward(layer.layers[t],dys[:,t,:])
        dhs[:,t,:] = dh
    end

    return dhs
end
function backward(layer::TimeMse_params,dloss)
    N, T, O = size(dloss)
    dys = zeros(Float64,(N,T,O))

    for t in 1:T
        dy = zeros(Float64,(N,O))
        dy = backward(layer.layers[t],0)
        dys[:,t,:] = dy
    end

    return dys
end


#TimeRNN functions
function set_state(layer::TimeRNN_params,h)
    layer.h = h    
end
function reset_state(layer::TimeRNN_params)
    layer.h = nothing
end
    
#==============
sample
===============#
mutable struct layer_a
    params
end

function layer_f(params)
    return layer_a(params)
end


end
