module Optimizer

#=====
params
======#
abstract type Optimizer_params end
mutable struct SGD_params <: Optimizer_params
    learning_rate::Float64 
end
mutable struct Adam_params <: Optimizer_params
    learning_rate::Float64
    p1
    p2
    e
    iter
end

#=========
コンストラクタ
==========#
function SGD(params::Array)
    return SGD_params(params[1])    
end
function Adam(params::Array)
    return Adam_params(params[1],params[2],params[3],params[4],0)
end

#=========
実行
==========#
function update(params::SGD_params,layers)
    for i in 1:2:length(layers)
        layers[i].W = layers[i].W -params.learning_rate*layers[i].dW
    end
end
function update(params::Adam_params,layers)
    # params.iter += 1
    # lr_t = params.learning_rate * sqrt.(1.0 .- params.p2.^params.iter) ./ (1.0 .- params.p1.^params.iter)
    for i in 1:2:length(layers)
        v = params.p1 * layers[i].v + (1-params.p1)*layers[i].dW
        s = params.p2 * layers[i].s + (1-params.p2)*(layers[i].dW.^2)
        layers[i].W = layers[i].W - params.learning_rate*(v./sqrt.(params.e .+s))
        layers[i].v = v
        layers[i].s = s

        # layers[i].v += (1 - params.p1) .* (layers[i].dW - layers[i].v)
        # layers[i].s += (1 - params.p2) .* (layers[i].dW.^2 - layers[i].s)

        # layers[i].W -= lr_t .* layers[i].v ./ (sqrt.(layers[i].s) .+ 10^(-7))
    end
end

end