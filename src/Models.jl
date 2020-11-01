module Models

include("Layers.jl")
include("Optimizer.jl")
include("DataSet.jl")

export MLP,predict,learn,dataset

using Random

#========
Model_params
========#
abstract type Model_params end
mutable struct MLP_params <:Model_params
    layers::Array
    debug::Bool
end

#==========
コンストラクタ
===========#
function MLP(;activation="Sigmoid",loss="CrossEntropy",neuron,debug=false)
    #neuronには入力，中間1，中間2，...中間n，出力までの各ニューロン数を格納

    if debug
        Random.seed!(123456)
    end

    #レイヤ初期化
    layers = []
    for i in 1:length(neuron)-1
        #パラメータ生成
        W = randn(neuron[i],neuron[i+1])
        b = randn(neuron[i+1])

        if i == length(neuron)-1
            #活性化関数&損失関数レイヤ
            activation_a = string("Layers.",activation,"_with_",loss,"()")
        else
            #活性化関数レイヤ
            activation_a = string("Layers.",activation,"()")
        end
        activation_layer = Meta.parse(activation_a)

        #レイヤ生成
        layers_tmp = [
            Layers.Affine(W,b),
            eval(activation_layer),
       ]

       append!(layers,layers_tmp)

    end

    return MLP_params(layers,debug)
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


#================
学習
================#
function learn(params::MLP_params;batch_size,epoch,learning_rate,data,t_data=[],optimizer="SGD")
    ite = size(data)[1] ÷ batch_size
    epoch_loss = 0
    #optimizer生成
    optimizer_a = string("Optimizer.$optimizer($learning_rate)")
    optimizer = eval(Meta.parse(optimizer_a))
    loss_list = [] #エポック毎の損失の平均を格納
    #1エポック=データを1巡
    for i in 1:epoch
        ite_avg_loss = 0 #1イテレーション毎の平均損失
        #データのインデックスシャッフル
        idx = shuffle(1:size(data)[1])
        data = data[idx,:]
        t_data = t_data[idx,:]

        for k in 1:ite
            batch_data = data[1+(k-1)*batch_size:k*batch_size,1:end]
            params.layers[end].t = t_data[1+(k-1)*batch_size:k*batch_size,1:end]
            loss = predict(params,batch_data)
            #逆伝播
            dout=0
            for j in length(params.layers):-1:1
                dout = Layers.backward(params.layers[j],dout)
            end
            ite_avg_loss += sum(loss)/length(loss)
        end

        #更新
        Optimizer.update(optimizer,params.layers)

        epoch_avg_loss = ite_avg_loss/ite #1エポック毎の平均損失
        append!(loss_list,epoch_avg_loss)
    end

    params.layers[end].t = []

    return loss_list
end

#================
サンプルデータ
=================#
function dataset(n::Int64)
    return DataSet.two_squared(n)
end

end