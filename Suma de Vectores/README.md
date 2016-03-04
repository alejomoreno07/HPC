# IMPLEMENTACION Y COMPARACION DE LA SUMA DE VECTORES DE FORMA SECUENCIAL Y  EN PARALELO



### Introducción

El proceso de la suma de vectores, es un proceso computacionalmente costoso, si se asume una implementación estandar con una complejidad `O(N)` en donde el escalamiento en cuanto a los datos posiblemente tienda a un crecimiento que pueda ser muy elevado, y que genere en últimas un tiempo de respuesta muy tardío.

Por tanto, se encuentra  la necesidad de una implementación que pueda reducir los tiempos de procesamiento y respuesta hasta que puedan ser considerablemente adecuados.

En medio de ello se realizará una implementación de la solución secuencial y otra paralela de dicho problema, para luego ejecutarlos con distintos tamaños de datos, en donde se tomarán sus tiempos de ejecución para su posterior comparación.

### Especificaciones técnicas

- Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz
- NVIDIA GPU Tesla K40c
- 16 GB RAM

**Implementacion secuencial:** Para la implementacion secuencial se utiliza `malloc` y `free`.

**Implementacion CUDA:** El mismo alojamiento para la implementacion secuencial + `cudaMalloc` y `cudaMemcpy` para manejar la memoria del dispositivo.
