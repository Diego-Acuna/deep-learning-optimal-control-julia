using Plots

# Visualizar resultados
function plot_results(X, y_classes, y_opt, num_layers)
    # Proyectar datos a 2D para graficar
    X_2D = X[:, 1:2]  # Solo usar las dos primeras características
    y_transformed = [y_opt[j][:, 1:2] for j in 1:num_layers+1]  # Transformaciones en 2D

    class_colors = [c == 0 ? "#696EEC" : :orange for c in y_classes]
    
    # Graficar datos originales
    scatter(X_2D[:, 1], X_2D[:, 2], color=class_colors, legend=false, title="Datos Originales")
    savefig("datos_originales.png")
    
    # Graficar transformaciones a lo largo de las capas
    for j in 1:num_layers+1
        scatter(y_transformed[j][:, 1], y_transformed[j][:, 2], color=class_colors, legend=false,
                title="Transformación Después de la Capa Neuronal $j")
        savefig("transformacion_capa_$j.png")
    end
end

# Llamar a la función
plot_results(X, y_classes, y_opt, num_layers)