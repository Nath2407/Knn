# Datos de entrenamiento de Iris con 4 características
data = [
    ([5.1, 3.5, 1.4, 0.2], 'setosa'),
    ([4.9, 3.0, 1.4, 0.2], 'setosa'),
    ([6.0, 2.2, 4.0, 1.0], 'versicolor'),
    ([6.1, 2.8, 4.7, 1.2], 'versicolor'),
    ([6.8, 3.2, 5.9, 2.3], 'virginica'),
    ([6.3, 3.3, 6.0, 2.5], 'virginica')
]

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

# Función de k-NN para clasificación
def k_nearest_neighbors(training_data, test_instance, k):
    distances = []

    for item in training_data:
        distance = euclidean_distance(test_instance, item[0])
        distances.append((item, distance))

    # Ordenar las distancias y obtener los k vecinos más cercanos
    distances.sort(key=lambda x: x[1])
    neighbors = [item for item, distance in distances[:k]]

    # Contar la frecuencia de cada clase en los vecinos cercanos
    class_counts = {}
    for neighbor in neighbors:
        label = neighbor[1]
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    # Determinar la clase más común en los vecinos cercanos
    max_class = max(class_counts, key=class_counts.get)
    return max_class

# Punto de prueba con las 4 características
test_point = [6.0, 3.0, 4.5, 1.5]

# Valor de k (número de vecinos cercanos)
k = 3

# Clasificar el punto de prueba usando k-NN
result = k_nearest_neighbors(data, test_point, k)
print(f'El punto de prueba pertenece a la clase: {result}')
