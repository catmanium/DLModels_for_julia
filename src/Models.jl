module Models

include("Layers.jl")
include("Optimizer.jl")
include("DataSet.jl")

export MLP,SimpleRNN,predict,learn,optimizer,dataset,model_f

using Random

#==============================
共通メンバ変数
・params : 各レイヤに重みをリスト化
・grads : 各レイヤに勾配をリスト化 → Layers.forward()でLayerのメンバに代入したら合わせて変更される
・layers : 各レイヤをリスト化
・loss_layer : 損失関数レイヤ
・optimizer : 最適化アルゴリズム

共通メソッド
・モデル名(;neuron,loss_layer) :コンストラクタ
    ・neuron : 入力～中間～出力　までのニューロン数を指定
    ・loss_layer : 損失関数（フルネーム指定）
        ・重み，バイアス，レイヤ，損失関数レイヤの初期化
        ・生成したレイヤからparams,gradsを取り出しメンバ変数へ保管

・predict(model_params;data) : 推論　→　スコア算出

・learn(model_params;batch_size,max_epoch,data,t_data=[]) 
    ※以下をmax_epoch回ループ,max_ite回ループ
    ・predict()実行　→　損失算出(loss_layerにt_dataが指定してあれば損失を返す)
    ・Model.layersを逆からbackward() → 勾配算出
    ・params更新


===============================#


#========
Model_params
========#
abstract type Model_params end
mutable struct MLP_params <:Model_params
    params
    grads
    layers::Array
    optimizer
    debug::Bool
end
mutable struct SimpleRNN_params <:Model_params
    params
    grads
    layers::Array
    rnn_layer
    optimizer
end

#==========
コンストラクタ
===========#
function MLP(;activation="Sigmoid",loss="Sigmoid_with_CrossEntropy",neuron,debug=false)
    #neuronには入力，中間1，中間2，...中間n，出力までの各ニューロン数を格納

    if debug
        Random.seed!(123456)
    end

    optimizer = []
    params = []
    grads = []

    #レイヤ初期化
    layers = []
    for i in 1:length(neuron)-1
        layers_tmp = []
        params_tmp = []
        grads_tmp = []
        #パラメータ生成
        W = randn(neuron[i],neuron[i+1])
        b = randn(neuron[i+1])
        dW = zeros(Float64,size(W))
        db = zeros(Float64,size(b))
        #各レイヤのparams,gradsをリストで管理
        params_tmp = [W,b]
        grads_tmp = [dW,db]
        append!(params,params_tmp)
        append!(grads,grads_tmp)

        if i == length(neuron)-1
            #活性化関数&損失関数レイヤ
            activation_a = string("Layers.",loss,"()")
        else
            #活性化関数レイヤ
            activation_a = string("Layers.",activation,"()")
        end
        activation_layer = Meta.parse(activation_a)

        #レイヤ生成
        layers_tmp = [
            Layers.Affine(params_tmp,grads_tmp),
            eval(activation_layer),
       ]

       append!(layers,layers_tmp)
    end

    return MLP_params(params,grads,layers,optimizer,debug)
end
function SimpleRNN(;data_size,hidden_size,output_size)
    #hidden_size RNNレイヤの隠れ状態の次元数
    D, H, O = data_size, hidden_size, output_size

    #重みの初期化
    rnn_Wx = randn(D,H)/sqrt(D)
    rnn_Wh = randn(H,H)/sqrt(H)
    rnn_b = zeros(H)
    affine_W = randn(H,O)/sqrt(H)
    affine_b = zeros(O)
    params = []
    grads = []

    #レイヤの初期化
    layers = [
        Layers.TimeRNN(rnn_Wx,rnn_Wh,rnn_b,true),
        Layers.TimeAffine(affine_W,affine_b),
        Layers.TimeMse()
    ]
    
    optimizer = []

    return SimpleRNN_params(params,grads,layers,layers[1],[])
end


#================
推論
================#
function predict(params::MLP_params,data)
    out = data
    for i in 1:length(params.layers)
        out = Layers.forward(params.layers[i],out)
    end
    
    #推論と学習に応じて，スコアか損失か変わる(学習データの有無で判別)
    return out
end
function predict(params::SimpleRNN_params,data)
    hs = Layers.forward(params.layers[1],data)
    ys = Layers.forward(params.layers[2],hs)
    return ys
end


#================
学習
================#
function learn(this::MLP_params;batch_size,max_epoch,data,t_data=[])
    ite = size(data)[1] ÷ batch_size
    epoch_loss = 0
    min_loss = 0
    min_loss_ep = 0
    loss_list = [] #エポック毎の損失の平均を格納
    #1エポック=データを1巡
    for i in 1:max_epoch
        ite_avg_loss_sum = 0 #1イテレーション毎の平均損失の総和
        #データのインデックスシャッフル
        idx = shuffle(1:size(data)[1])
        data = data[idx,:]
        t_data = t_data[idx,:]
        epoch_avg_loss = 0

        for k in 1:ite
            batch_data = data[1+(k-1)*batch_size:k*batch_size,1:end]
            this.layers[end].t = t_data[1+(k-1)*batch_size:k*batch_size,1:end]
            #順伝播
            loss = predict(this,batch_data)
            #逆伝播
            dout=1
            for j in length(this.layers):-1:1
                dout = Layers.backward(this.layers[j],dout)
            end
            ite_avg_loss_sum += sum(loss)/length(loss) #1イテレーションの平均損失を足していく

            #更新
            Optimizer.update(this.optimizer,this)
        end

        epoch_avg_loss = ite_avg_loss_sum/ite #1エポック毎の平均損失

        #10回に一回出力する
        if i%(max_epoch/10) == 0
            println("ep.$i : Loss :　",epoch_avg_loss)
        end

        if i==1 || min_loss > epoch_avg_loss
            min_loss = epoch_avg_loss
            min_loss_ep = i
        end

        append!(loss_list,epoch_avg_loss)
    end

    println("min_loss : $min_loss")
    println("min_loss_ep : $min_loss_ep")

    #predict単体で実行するために初期化しておく
    this.layers[end].t = []

    return loss_list
end
function learn(params::SimpleRNN_params;batch_size,max_epoch,window_size,data,t_data=[])
    X = size(data,1) #総データ数
    D = size(data,2) #データ次元数
    T = window_size #RNNレイヤ数
    N = batch_size #バッチ数

    max_ite = (X/N)*(1/T) #イテレーション数
    loss_list = [] #avg_lossのリスト

    for epoch in 1:max_epoch
        total_loss = 0 #損失合計
        avg_loss = 0 #1エポックの平均損失
        for ite in 1:max_ite
            #ミニバッチ作成
            xs = zeros(Float64,(N,T,D))
            ts = zeros(Float64,(N,T,D))
            for n in 1:N
                st = 1+(T*(ite-1))+(X/N)*(n-1)
                ed = T+(T*(ite-1))+(X/N)*(n-1)
                xs[Int(n),:,:] = data[Int(st):Int(ed),:]
                ts[Int(n),:,:] = t_data[Int(st):Int(ed),:]
            end
            #順伝播
            ys = predict(params,xs)
            loss = Layers.forward(params.layers[end],ys,ts)
            if epoch==1 && ite==1
                println(size(loss))
            end
            total_loss += sum(loss)
            #逆伝播
            dl = ones(Float64,size(loss))
            dys = Layers.backward(params.layers[end],dl)
            dhs = Layers.backward(params.layers[2],dys)
            dxs = Layers.backward(params.layers[1],dhs)
            #更新
        end
        avg_loss = total_loss/(T*D*N)
        append!(loss_list,avg_loss)
    end


    return loss_list
end

#================
最適化 optimizer
=================#
function optimizer(this::Model_params;name="Adam",learning_rate=0.001,p1=0.95,p2=0.99,e=10^(-12))
    vs = []
    ss = []
    for i in 1:length(this.params)
        v = zeros(Float64,size(this.params[i]))
        s = zeros(Float64,size(this.params[i]))
        append!(vs,[v])
        append!(ss,[s])
    end
    hyper_params = [
        learning_rate,
        p1,
        p2,
        e,
        vs,
        ss
    ]
    #optimizer生成
    optimizer_a = string("Optimizer.$name($hyper_params)")
    this.optimizer = eval(Meta.parse(optimizer_a))
end

#================
サンプルデータ
=================#
function dataset(type::String,n::Int64)
    type_name = string("DataSet.",type,"($n)")
    type_function = Meta.parse(type_name)
    return eval(type_function)
end


#==============
sample
===============#
mutable struct model_a
    params
    layers
end

function model_f()
    params = []
    layers = []
    for i in 1:4
        layers_tmp = []
        params_tmp = []
        W = randn(2,3)
        b = randn(4)
        #各レイヤのparams,gradsをリストで管理
        params_tmp = [W,b]
        append!(params,params_tmp)
        layers_tmp = [
            Layers.layer_f(params_tmp),
        ]
        append!(layers,layers_tmp)
    end
    
    return model_a(params,layers)
end


end