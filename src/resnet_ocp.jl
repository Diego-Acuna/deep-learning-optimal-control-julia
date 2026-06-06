using JuMP
using Ipopt
using LinearAlgebra
using Statistics

# Consideramos la función de activación ReLU para
# la red neuronal
function activation(x)
    return max.(0.0, x)
end

# Considerar la derivada de ReLU como la indicatriz de
# que x sea positivo traía errores a la hora de llamarla en
# las constraints del problema de optimización vía JuMP, por 
# lo que se consideró esta aproximación suave de la derivada
function activation_derivative(x)
    return 1.0 ./ (1.0 .+ exp.(-10 * x)) 
end

function resnet_ocp_hamiltonian(X, y_target, num_layers, Δt)
    n_samples, n_features = size(X)

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", 1e-6)

    # Se definen las variables de estado y, del control u=(K,β)
    # y del estado adjunto p
    @variable(model, y_states[1:num_layers+1, 1:n_samples, 1:n_features])
    @variable(model, K_controls[1:num_layers, 1:n_features, 1:n_features])
    @variable(model, β_controls[1:num_layers, 1:n_features])
    @variable(model, p_adjoints[1:num_layers+1, 1:n_samples, 1:n_features])

    # Inicializar el estado inicial para todas las muestras
    for i in 1:n_samples
        for k in 1:n_features
            @constraint(model, y_states[1, i, k] == X[i, k])
        end
    end

    # Dinámica discreta de ResNet
    for j in 1:num_layers
        for i in 1:n_samples
            @constraint(
                model,
                y_states[j+1, i, :] .== y_states[j, i, :] + Δt * activation(K_controls[j, :, :] * y_states[j, i, :] .+ β_controls[j, :])
            )
        end
    end

    # Condiciones para los adjuntos (ecuaciones adjuntas)
    for j in num_layers:-1:1
        for i in 1:n_samples
            @constraint(
                model,
                p_adjoints[j, i, :] .== p_adjoints[j+1, i, :] - Δt * (K_controls[j, :, :] * (p_adjoints[j+1, i, :] .* activation_derivative(K_controls[j, :, :] * y_states[j, i, :] .+ β_controls[j, :])))
            )
        end
    end

    # Condición final para los adjuntos
    for i in 1:n_samples
        @constraint(
            model,
            p_adjoints[num_layers+1, i, :] .== 2 * (y_states[num_layers+1, i, :] .- y_target[i])
        )
    end

    # Función de costo basada en el Hamiltoniano
    @objective(
        model,
        Min,
        sum((y_states[end, :, 1] .- y_target).^2)
    )

    optimize!(model)

    K_opt = [value.(K_controls[j, :, :]) for j in 1:num_layers]
    β_opt = [value.(β_controls[j, :]) for j in 1:num_layers]
    y_opt = [value.(y_states[j, :, :]) for j in 1:num_layers+1]
    p_opt = [value.(p_adjoints[j, :, :]) for j in 1:num_layers+1]
    v_opt = objective_value(model)

    return K_opt, β_opt, y_opt, p_opt, v_opt
end

# Generar datos sintéticos para clasificación binaria
function generate_binary_data(n_samples, n_features)
    X = randn(n_samples, n_features)
    X = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    true_weights = randn(n_features)
    y = X * true_weights .+ 0.1 * randn(n_samples)  # Agregar ruido
    y_classes = y .> median(y)  # Convertir a clases binarias
    return X, y_classes
end

# Configuración inicial
n_samples = 50
n_features = 5
num_layers = 34
Δt = 0.01  # Paso temporal

X, y_classes = generate_binary_data(n_samples, n_features)

# Resolver el problema de control óptimo
y_target = y_classes .* 1.0  # Convertir etiquetas binarias a flotantes
K_opt, β_opt, y_opt, p_opt, v_opt = resnet_ocp_hamiltonian(X, y_target, num_layers, Δt)

println("Problema resuelto con JuMP: ", v_opt)
println("Pesos óptimos (última capa): ", K_opt[end])
println("Sesgos óptimos (última capa): ", β_opt[end])
println("Estados óptimos (última capa): ", y_opt[end])
println("Adjuntos óptimos (última capa): ", p_opt[end])