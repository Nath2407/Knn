import math

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

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
        label = neighbor[0][1]
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    # Determinar la clase más común en los vecinos cercanos
    max_class = max(class_counts, key=class_counts.get)
    return max_class

# Ejemplo de datos de entrenamiento
training_data = [([2, 3], 'A'), ([5, 4], 'B'), ([9, 6], 'A'), ([4, 7], 'B')]

# Punto de prueba
test_point = [6, 5]

# Valor de k (número de vecinos cercanos)
k = 2

# Clasificar el punto de prueba usando k-NN
result = k_nearest_neighbors(training_data, test_point, k)
print(f'El punto de prueba pertenece a la clase: {result}')
