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

その他
・各レイヤの勾配はコンストラクタ内で生成 -> Model,Timeレイヤと非共有にするため
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
    params #Wx,Wh,b
    grads #dWx,dWh,db
    cache #x, h_prev, h_next
    padding #空白データ，TimeRnn，Modelとメモリ共有
end
mutable struct TimeRNN_params <: Params
    params #Wx,Wh,b
    grads #dWx,dWh,db
    layers #複数のRNNレイヤを管理
    h #次のエポックに引き継ぐ
    dh
    stateful
    padding #空白データ，Rnn，Modelとメモリ共有
end
mutable struct TimeAffine_params <: Params
    params #W,b
    grads #dW
    layers #複数のAffineレイヤを管理
end
mutable struct TimeMse_params <: Params
    layers
end
mutable struct LSTM_params <: Params
    params #Wx,Wh,b
    grads #dWx,dWh,db
    cache #x, h_prev, h_next,i,f,g,o,c_next
    padding #空白データ，TimeRnn，Modelとメモリ共有
end
mutable struct TimeLSTM_params <: Params
    params #Wx,Wh,b
    grads #dWx,dWh,db
    layers #複数のRNNレイヤを管理
    h #次のエポックに引き継ぐ
    c
    dh
    stateful
    padding #空白データ，Rnn，Modelとメモリ共有
end


#レイヤコンストラクタ
function Sigmoid()
    out = []
    params = []
    grads = []
    return Sigmoid_params(params,grads,out)
end
function Affine(W::Array{Float64,2},b::Array{Float64,2})
    params = [W,b]
    dW = zeros(Float64,size(W))
    db = zeros(Float64,size(b))
    grads = [dW,db]
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
function RNN(Wx::Array{Float64,2},Wh::Array{Float64,2},b::Array{Float64,2},padding)
    #勾配の初期化
    dWx = zeros(size(Wx))
    dWh = zeros(size(Wh))
    db = zeros(size(b))
    #パラメータ，勾配のリスト化
    params = [Wx,Wh,b]
    grads = [dWx,dWh,db]
    cache = []
    return RNN_params(params,grads,cache,padding)
end
function TimeRNN(Wx::Array{Float64,2},Wh::Array{Float64,2},b::Array{Float64,2},stateful::Bool,padding)
    #勾配初期化
    dWx = zeros(size(Wx))
    dWh = zeros(size(Wh))
    db = zeros(size(b))
    #重み，バイアス，勾配をリスト化
    params = [Wx,Wh,b]
    grads = [dWx,dWh,db]
    #その他
    layers = nothing
    dh = nothing
    h = nothing
    return TimeRNN_params(params,grads,layers,h,dh,stateful,padding)
end
function TimeAffine(W::Array{Float64,2},b::Array{Float64,2})
    dW = zeros(size(W))
    db = zeros(size(b))
    params = [W,b]
    grads = [dW,db]
    layers = []
    return TimeAffine_params(params,grads,layers)
end
function TimeMse()
    return TimeMse_params([])
end
function LSTM(Wx::Array{Float64,2},Wh::Array{Float64,2},b::Array{Float64,2},padding)
    #勾配の初期化
    dWx = zeros(size(Wx))
    dWh = zeros(size(Wh))
    db = zeros(size(b))
    #パラメータ，勾配のリスト化
    params = [Wx,Wh,b]
    grads = [dWx,dWh,db]
    cache = []
    return LSTM_params(params,grads,cache,padding)
end
function TimeLSTM(Wx::Array{Float64,2},Wh::Array{Float64,2},b::Array{Float64,2},stateful::Bool,padding)
    #勾配初期化
    dWx = zeros(size(Wx))
    dWh = zeros(size(Wh))
    db = zeros(size(b))
    #重み，バイアス，勾配をリスト化
    params = [Wx,Wh,b]
    grads = [dWx,dWh,db]
    #その他
    layers = nothing
    dh = nothing
    h = nothing
    c = nothing
    return TimeLSTM_params(params,grads,layers,h,c,dh,stateful,padding)
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
    Wx,Wh,b = layer.params
    padding = layer.padding

    t = h_prev*Wh + x*Wx .+ b
    h_next = tanh.(t)

    #x に padding が指定してある場合，h_prevの値をそのまま伝播
    if padding !== nothing
        #インデックス取得
        ids = Tuple.(findall(x->x==padding,x))
        #行番号を取得
        id_cs = first.(ids)
        max,min = extrema(id_cs)
        #h_next に h_prevを代入
        h_next[max:min,:] .= h_prev[max:min,:]
    end

    layer.cache = [x,h_prev,h_next]

    return h_next
end
function forward(layer::TimeRNN_params,xs)
    Wx,Wh,b = layer.params
    #バッチサイズ，ブロック数，入力データ数
    N, T, D = size(xs)
    D, H = size(Wx)

    layer.layers = []
    hs = zeros(Float64,(N,T,H))

    if !layer.stateful || layer.h === nothing
        layer.h = zeros(Float64,(N,H))
    end

    for t in 1:T
        rnn_layer = RNN(Wx,Wh,b,layer.padding)
        layer.h = forward(rnn_layer,xs[:,t,:],layer.h)
        hs[:,t,:] = layer.h
        push!(layer.layers,rnn_layer)
    end

    return hs
end
function forward(layer::TimeAffine_params,hs)
    W,b = layer.params
    dW = layer.grads
    N,T,H = size(hs)
    H,O = size(W)
    ys = zeros(Float64,(N,T,O))

    for t in 1:T
        h = hs[:,t,:]
        affine_layer = Affine(W,b)
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
function forward(layer::LSTM_params,x,h_prev,c_prev)
    Wx, Wh, b = layer.params
    N, H = size(h_prev)

    A = x * Wx + h_prev * Wh .+ b

    #slice
    f = A[:, begin:H]
    g = A[:, H+1:2*H]
    i = A[:, 2*H+1:3*H]
    o = A[:, 3*H+1:end]

    f = 1.0 ./ (1.0 .+ exp.(-f)) #Sigmoid
    g = tanh.(g)
    i = 1.0 ./ (1.0 .+ exp.(-i))
    o = 1.0 ./ (1.0 .+ exp.(-o))

    c_next = f .* c_prev + g .* i
    h_next = o .* tanh.(c_next)

    layer.cache = [x, h_prev, c_prev, i, f, g, o, c_next]

    return h_next, c_next

end
function forward(layer::TimeLSTM_params,xs)
    Wx,Wh,b = layer.params
    #バッチサイズ，ブロック数，入力データ数
    N, T, D = size(xs)
    H = size(Wh,1)

    layer.layers = []
    hs = zeros(Float64,(N,T,H))

    if !layer.stateful || layer.h === nothing
        layer.h = zeros(Float64,(N,H))
    end

    if !layer.stateful || layer.c === nothing
        layer.c = zeros(Float64,(N,H))
    end

    for t in 1:T
        rnn_layer = LSTM(Wx,Wh,b,layer.padding)
        layer.h, layer.c = forward(rnn_layer,xs[:,t,:],layer.h,layer.c)
        hs[:,t,:] = layer.h
        push!(layer.layers,rnn_layer)
    end

    return hs
end

#backward
function backward(layer::Affine_params,din)
    layer.grads[1] .= (layer.in)' * din #dW
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
    Wx, Wh, b = layer.params
    x, h_prev, h_next = layer.cache

    dt = dh_next .* (1 .- h_next .^2)
    db = sum(dt,dims=1)
    dWh = h_prev' * dt
    dh_prev = dt * Wh'
    dWx = x' * dt
    dx = dt * Wx'

    layer.grads .= [dWx,dWh,db]

    return dx,dh_prev
end
function backward(layer::TimeRNN_params,dhs)
    Wx, Wh, b = layer.params
    D, H = size(Wx)
    
    #dhsの次元数から処理を分岐
    if ndims(dhs) == 2
        #many to one (dhs が (N,H))
        N, H = size(dhs)
        T = length(layer.layers)
        dxs = zeros(Float64,(N,T,D))
        #代入する勾配
        grads = copy(layer.grads)
        grads -= grads #全て0に
        
        dh = copy(dhs)
        for t in T:-1:1
            rnn_layer = layer.layers[t]
            dx, dh = backward(rnn_layer,dh)
            dxs[:,t,:] = dx

            #各レイヤの勾配合算 
            grads += rnn_layer.grads
        end

        #TimeRNN,Modelsの勾配更新-> Model.gradsとメモリ共有しているため同時に更新される
        layer.grads[1][:] = grads[1][:]
        layer.grads[2][:] = grads[2][:]
        layer.grads[3][:] = grads[3][:]

    elseif ndims(dhs) == 3
        #many to many (dhs が (N,T,H))
        N, T, H = size(dhs)
        dxs = zeros(Float64,(N,T,D))

        dh = zeros(Float64,size(dhs[:,1,:]))
        for t in T:-1:1
            rnn_layer = layer.layers[t]
            dx, dh = backward(rnn_layer,dhs[:,t,:]+dh)
            dxs[:,t,:] = dx

            #勾配合算 -> Model.gradsとメモリ共有しているため同時に更新される
            for i in 1:3
                layer.grads[i] .+= rnn_layer.grads[i]
            end
        end
    end

    layer.dh = dh

    return dxs
    
end
function backward(layer::TimeAffine_params,dys)
    N, T, O = size(dys)
    H, O = size(layer.layers[1].params[1]) #W
    dhs = zeros(Float64,(N,T,H))

    for t in 1:T
        dh = zeros(Float64,(N,H))
        affine_layer = layer.layers[t]
        dh = backward(affine_layer,dys[:,t,:])
        dhs[:,t,:] = dh
        #勾配合算 -> Model.gradsとメモリ共有しているため同時に更新される
        for i in 1:2
            layer.grads[i] .+= affine_layer.grads[i]
        end
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
function backward(layer::LSTM_params,dh_next,dc_next)
    Wx, Wh, b = layer.params
    x, h_prev, c_prev, i, f, g, o, c_next = layer.cache

    tanh_c_next = tanh.(c_next)

    ds = dc_next + (dh_next .* o) .* (1 .- tanh_c_next.^2)

    dc_prev = ds .* f

    di = ds .* g
    df = ds .* c_prev
    d_o = dh_next .* tanh_c_next
    dg = ds .* i

    di .*= i .* (1 .- i)
    df .*= f .* (1 .- f)
    d_o .*= o .* (1 .- o)
    dg .*= (1 .- g.^2)

    dA = hcat(df,dg,di,d_o)

    dWh = h_prev' * dA
    dWx = x' * dA
    db = sum(dA,dims=1)

    layer.grads[1][:] = dWx
    layer.grads[2][:] = dWh
    layer.grads[3][:] = db

    dx = dA * Wx'
    dh_prev = dA * Wh'

    return dx, dh_prev, dc_prev
end
function backward(layer::TimeLSTM_params,dhs)
    Wx, Wh, b = layer.params
    D, H = size(Wx)
    
    #dhsの次元数から処理を分岐
    if ndims(dhs) == 2
        #many to one (dhs が (N,H))
        N, H = size(dhs)
        T = length(layer.layers)
        dxs = zeros(Float64,(N,T,D))
        #代入する勾配
        grads = copy(layer.grads)
        grads -= grads #全て0に
        
        dh = copy(dhs)
        dc = zeros(Float64,size(dh))
        
        for t in T:-1:1
            rnn_layer = layer.layers[t]
            dx, dh, dc = backward(rnn_layer,dh,dc)
            dxs[:,t,:] = dx

            #各レイヤの勾配合算 
            grads += rnn_layer.grads
        end

        #TimeRNN,Modelsの勾配更新-> Model.gradsとメモリ共有しているため同時に更新される
        layer.grads[1][:] = grads[1][:]
        layer.grads[2][:] = grads[2][:]
        layer.grads[3][:] = grads[3][:]

    elseif ndims(dhs) == 3
        #many to many (dhs が (N,T,H))
        N, T, H = size(dhs)
        dxs = zeros(Float64,(N,T,D))

        dh = zeros(Float64,size(dhs[:,1,:]))
        dc = zeros(Float64,size(dhs[:,1,:]))
        for t in T:-1:1
            rnn_layer = layer.layers[t]
            dx, dh, dc = backward(rnn_layer,dhs[:,t,:]+dh, dc)
            dxs[:,t,:] = dx

            #勾配合算 -> Model.gradsとメモリ共有しているため同時に更新される
            for i in 1:3
                layer.grads[i] .+= rnn_layer.grads[i]
            end
        end
    end

    layer.dh = dh

    return dxs
    
end


#TimeRNN functions
function set_state(layer::TimeRNN_params,h)
    layer.h = h    
end
function reset_state(layer::TimeRNN_params)
    layer.h = nothing
end

function reset_state(layer::TimeLSTM_params)
    layer.h = nothing
    layer.c = nothing
end

end
