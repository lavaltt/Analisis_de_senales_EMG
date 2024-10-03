# Analisis de senales EMG
## Descripción
Este proyecto realiza un análisis detallado de una señal fisiológica (EMG) obtenida del bíceps durante contracciones musculares. Incluye la visualización de la señal, filtrado  para eliminar el ruido fuera del rango útil, segmentación de la señal en ventanas alrededor de las contracciones detectadas, y un análisis espectral. Además, se realiza una prueba de hipótesis basada en la frecuencia mediana de cada segmento para evaluar la variabilidad entre contracciones, cuando el músculo se encuentra mpas cerca del fallo. 

## Adquisición de la señal EMG
Para la captura de la señal electromiográfica, se determino inicialmente el músculo a estudiar. Para el proyecto, se eligió el bíceps, teniendo en cuenta que  presenta un mejor muestreo en estudios de electromiografía (EMG), lo que facilita la obtención de la señal. Sus ventajas se deben a varias razones, pero una de las principales es su  gran tamaño y definición, simplificando el proceso de ubicación de los electrodos [^1^]. Una vez determinado lo anterior, se ha decidido realizar la adquisición por medio de una DAQ (Data Acquisition System) de National Instruments, que es un dispositivo utilizado para medir señales físicas y convertirlas en datos digitales que puedan ser procesados. 

<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/daq.jpg?raw=true"  width="400" height="300">

*Figura 1: Data Acquisition System. Tomado de : [^2^]*

Adicionalmente, se utilizó un módulo AD8232, sensor que permite la medición de la actividad muscular,si bien se destaca por su precisión en señales cardíacas, el procesos de amplificación y captura que realiza resulta útil para cualquier señal muscular y permite la  lectura por parte de la DAQ.

<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/modulo.jpg?raw=true"  width="400" height="300">

*Figura 2: Modulo de adquisición y amplificación AD8832. Tomado de : [^3^]*

Inicialmente, se conectaron los electrodos en la posición indicada en la figura 3 a un sujeto de prueba y se le pidio realizar el movimiento de flexión y extensión del codo, sosteniendo el pso de una botela de agua de un litro. El movimiento fue realizado lento pero contante, asegurando las contracciones y que el músculo del sujto llegase a fallo. Los electrodos conectados al modulo enviaban la señal para que esta pudiera ser amplificada. 
Una vez amplificada, la señal llega a la DAQ, que realiza un muestreo a una frecuencia determinada (en este caso 300 Hz) y entrega los valores digitalizados para su almacenamiento y posterior análisis en Python.
Además de ello, se conectó la salida de la DAQ al osciloscopio para verificar que la señal visualizada en ppython no presentara cambios significativos en realcióna la observada en el osciloscopio. 

![Electrodos](https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/musculoelectrodos.jpg?raw=true)

*Figura 3: Ubicación de los electrodos para la adquisición de la señal del bíceps. Tomado de : [^4^]*


<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/circuito.jpg?raw=true"  width="400" height="300">

*Figura 4: Circuito realizado para la adquisicón de la señal. Tomado de : Autoría propia*


* Vídeo de la adquisición de la señal:

https://github.com/user-attachments/assets/dbd16faf-8960-4f84-946c-25c5915ed8ff



### DAQ y Python 
La DAQ fue enlazada por medio de código a python, de manera que pudiera visualizarse la captura de la señal en tiempo real y los datos fueran guardados en cualquier formato deseado. El código configura los parametros y la tarea de adquisicion, as u vez que estructura la graficación en vivo de la señal adquirida. (Código adjunto en el proyecto).

Muestra del código para la gráfica en tiempo real:
```python
 # Inicializar variables para la gráfica en tiempo real
    plt.ion()  # Habilitar modo interactivo
    fig, ax = plt.subplots()
    ts = int(sample_r * duracion)  # Total de muestras a adquirir
    x = np.arange(0, ts)  # Eje X
    y = np.zeros(ts)  # Inicializa los datos Y
    line, = ax.plot(x, y)
    ax.set_ylim(0, 3.5)  
    ax.set_xlim(0, total_samples)  
    ax.set_xlabel('Muestras')
    ax.set_ylabel('Voltaje (V)')
    ax.set_title('Señal en Tiempo Real')
```

## Carga de datos
Los datos adquiridos gracias a la DAQ y python, fueron guardados en un archivo tipo .csv para su correcto análisis y desarrollo. Inicialmente, para cargar los datos en este formato se utilizó la libreria pandas y se graficaron los adtos para observar la señal EMG obtenida. 

```python
import pandas as pd

# Cargar los datos
datos = pd.read_csv(r'C:\Users\Automatizacion 2\Downloads\V\Señales\lab_señales\voltajes_señal3.csv')
EMG = datos['Voltaje (V)'] 
```
<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/orgiginalsignal.png?raw=true"  width="1000" height="500">

*Figura 5: Señal original adquirida. Tomado de : Autoría propia*

Dicha señal cuenta con las siguientes características:
* Frecuencia de muestreo (fs) = 300Hz
* Tiempo de muestreo (ts) = 1200 ms
* Contracciones = La señal no es muy clara en cuanto a la cantidad de contracciones realizadas, por lo que se tomo como base el análisis y procesamiento de la misma para determinar el número, ya que durante la captura no se realizó un conteo. Teniendo en cuenta lo anterior, la cantidad de contraccines fueron apróximadamente 14. 

La frecuencia de muestreo fue definida a partir del criterio de nyquist que indica que la fs debe ser más del doble de la frecuencia máxima, en este caso del músculo que estabamos midiendo. De acuerdo con Moreno Sanz, la frecuencia del músculo se encuentra entre 50 – 100 Hz. [^5^]. En este caso particular, se tomó como frecuencia máxima 100 Hz. Se decidió tomar la fs como tres veces la frceuencia máxima, con el fin de evitar el riesgo de aliassing que puede ocurrir si se esta muy cerca del criterio de nyquist. El aliassing es el efecto que tienen las señales continuas en el tiempo de tornarse indistingibles al muestrearse digitalmente. [^6^].

El tiempo de muestreo resultó a partir del desgaste muscular y la tolerancia del sujeto, se logró muestrear apróximadamente 2 minutos. 

En cuanto a las contracciones, debido al ritmo en que se realizó el ejercicio (lento pero continuo), y al corto tiempo muestreado, se lograron apróximadamente 14. 

## Filtrado de la señal
Los filtros son herramientas fundamentales en el procesamiento de señales porque permiten modificar o eliminar ciertas partes de una señal, ya sea para mejorar su calidad, eliminar ruido, o extraer información relevante. En sistemas de adquisición de señales digitales, como los DAQ, el filtrado es crucial para evitar el aliasing, que ocurre cuando las frecuencias de la señal son más altas que la mitad de la frecuencia de muestreo, estando muy ecrca del teorema de Nyquist, lo que puede producir distorsiones en la señal.Para este caso, se decidió realizar un filtro pasabanda tipo butterworth. El filtro pasabanda permite el paso de un rango específico de frecuencias, lo que resulta esencial en procesamiento de señales EMG, dado que las frecuencias tanto muy altas, como muy bajas resultan ser ruidos e interferencias de los métodos de adquisición. Se eligió como frecuencias de corte 20 y 100 HZ, dado lo contenido en la literaturatura respecto a las frecuencias medias de la actividad muscular. [^5^]. 

Adicionalmente, teniendo en cuenta los parametros de diseño para los filtros, se eligió como frecuencias para atenuación en -20dB, 5 y 120 Hz. Estas frecuencias determinan la pendiente del filtro y el comienzo de la atenuación. 
En cuanto a la elección del tipo butterwoth, se determinó debido a su respuesta en frecuencia plana en la banda pasante, lo que facilita la eliminación de ruidos y artefactos sin introducir ningún tipo de ondulación que pueda afectar la interpretación de los datos. 

Para implementar el filtro en python y aplicarlo a ala señal es necesario conocer el orden del mismo. Este fue determinado de la siguiente manera:

Se deseaba diseñar un filtro pasabanda con -3.01 dB de atenuación en 20Hz y 100Hz, y con -20dB de atenuación en 5Hz y 120Hz. Inicialmente fue necesario hacer el proceso de conversión de unidades de Hz a radianes/s, de la siguiente manera:

![conversiones](https://github.com/user-attachments/assets/73bdcc99-18e5-4d11-afd2-4c94da59a45a)

Adicionalmente, se hizo uso de la transformación del filtro pasabajo a pasabanda para determinar la frecuencia de atenuación en -20dB para el diseño de un filtro pasabajo, esto teniendo en cuenta que resulta más sencillo diseñar este tipo de filtros y luego realizar la correspondiente transformación. 


![transformacion](https://github.com/user-attachments/assets/c00567f6-772a-4ebd-8adb-980a54bd85eb)

Los cálculos realizados a mano, fueron los siguientes:

![calculos](https://github.com/user-attachments/assets/175534c2-0473-4518-92bc-cab4f7900ac0)


Como se puede apreciar en el proceso matemático, el orden del filtro dio como resultado 4.28, sin embargo se debe aproximar al entero más cercano hacia arriba, por lo que se determina de orden 5. 

```python
fs = 300.0  # frecuencia de muestreo

lc = 20.0   # frecuencia de corte baja
hc = 100.0  # frecuencia de corte alta
n = 5       # orden del filtro

# Coeficientes del filtro Butterworth
b, a = signal.butter(n, [low, high], btype='band')
```
Posteriormente se aplicó el filtro a la señal y se gráfico obteniendo lo siguiente:

<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/filtersignal.png?raw=true"  width="1000" height="500">

*Figura 6: Señal original y señal filtrada con un butterworth pasabanda. Tomado de : Autoría propia*


## Aventanamiento de la señal

El aventanamiento de la señal (windowing), es una técnica utilizada para dividir o seccionar la señal en pequeños fragmentos conocidos comúnmente como ventanas con el fin de analizarla localmente en el tiempo, la idea es aplicar una función que selecciona una porción específica de la señal, para ir atenuando progresivamente los bordes hasta lograr reducir los efectos no deseados como las discontinuidades. Dicho método es fundamental para poder realizar un buen análisis espectral en este tipo de señales. 

Generalmente se suele utilizar cuando se necesita trabajar una señal en el dominio de la frecuencia, como en la transformada de fourier en la que que se asume la señal como periódica o infinita. Sin embargo la mayoría de señales adquiridas, como lo es este caso, son finitas. Uno de los mayores problemas durante la adquisición es el efecto ventana, un error que ocurre cuando no se reduce correctamente a 0 en los bordes; para mitigar este efecto se utilizan las funciones de aventanamiento antes de procesar la señal.	

Dentro de las ventajas del uso de esta técnica se encuentran,  la  mejora del análisis espectral, que reduce los errores y problemas en los bordes producidos por la transformada de fourier, así como el análisis de porciones específicas de la señal, lo que es muy útil para observar las características de la señal a lo largo del tiempo.

Los tipos  de aventanamiento mas usados son hanning y hamming, la primera utiliza la función coseno para atenuar los bordes y reducir las fugas espectrales, la segunda siendo muy similar a la primera posee una forma ligeramente diferente, pues no llega a cero en los extremos, resultando ideal para ciertos tipos de análisis . 

Hanning se utiliza para mejorar la resolución en el análisis de frecuencia, especialmente en señales EMG, razón principal para usarse en este caso ya que suaviza los bordes de la señal, lo que reduce las discontinuidades al calcular la transformada en el nálisis espectral. Se elige por su capacidad para ofrecer una transición suave, evitando picos indeseados que pueden afectar la interpretación de la señal. Al aplicar la ventana, la señal original se multiplica por el perfil de la ventana, suavizando las variaciones bruscas. que es lo deseado.

Para su implementación en la señal se debe realizar una detección de picos, que permita encontrar contracciones musculares en la señal. Esto debido a que el aventanamiento debe realizarse para cada contracción presente en la señal, es decir, para este caso se cuenta con 14 ventanas, por las 14 contracciones. Utiliza la función find_peaks de scipy.signal  y se asegura que haya al menos 0.5 segundos entre picos, para evitar detectar múltiples dentro de una misma contracción. A partir de ello se pueden definir los parametros de las ventanas y aplicarlas a la señal. 

```python
picos, _ = signal.find_peaks(datos_filtrados, height=threshold, distance=fs * 0.2)

ventana_size = int(0.2 * fs) # 200 ms antes y después de cada contracción
ventana_total = 2 * ventana_size

ventana_hanning = windows.hann(ventana_total)
segmento = datos_filtrados[inicio:fin] * ventana_hanning
```
Las ventanas fueron creadas por medio de la libreria scipy.signal, que contiene una función windows que permite hacer diferentes tipos de ventanas. En este caso la ventaan se hace con windows.hann, que se encarga de hacer una ventana tipo hanning. 

La ventana que se esta aplicando en cada segmento en el que se detecta un pico, se puede apreciar a continuación:

<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/ventanaaa.png?raw=true"  width="1000" height="500">

*Figura 7: Ventana aplicada. Tomado de : Autoría propia*

Dicha ventana tiene una longitud de 0.20 segundos, determinada en los parámetros de diseño para que tome este tiempo en los segmentos detectados como picos. 

En relación a los segmentos elegidos, la gráfica de una de las ventanas comparandola con la contracción sería la siguiente: 


<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/ventana%20y%20segmento1.png?raw=true"  width="1000" height="500">

*Figura 8: Segmento  y ventana de hanning. Tomado de : Autoría propia*

La gráfica azul representa el segmento detectado como un pico en la señal filtrada, sección a la que se aplica la ventana mostrada en la gráfica de color verde. 

La señal filtrada, con cada segmento ventaneado y la ventana que se le aplica a cada uno se puede observar a continuación: 

<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/se%C3%B1al%20con%20ventanas.png?raw=true"  width="1000" height="500">

*Figura 9: Señal aventanada. Tomado de : Autoría propia*

En esta graficación se pueden apreciar los picos detectados y la ventana que se le aplica a cada uno. Si se tiene en cuenta la señal filtrada en color rosado, se puede notar que algunos picos que podrian considerarse contracciones no fueron segmentados, esto puede deberse a complicaciones en la captura de la señal, que sigue sin ser muy clara respecto a ciertas secciones y también al diseño de las ventanas. Al asegurarse de que haya una distancia entre segmentos de 0.5 segundos, aquellos que estan muy cercanos no logran ser tomados. 

Por último, se hizo una reconstrucción de la señal a partir de los segmentos aventanados:

<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/se%C3%B1al%20reconstruida.png?raw=true"  width="1000" height="500">

*Figura 10: Señal reconstruida. Tomado de : Autoría propia*

Esta reconstrucción permite observar cada segmento ventaneado de manera conjunta, eliminando el resto d la señal. En terminos generales resulta precisa en cuanto a cómo debe verse una señal EMG capturada por equipo especializado. 

## Análisis espectral 
Como se mencionó en la sección de aventamiento, dicho proceso se realiza principalmente para poder aplicar un análisis espectral por medio de la tranformada de Fourier. Este análisis espectral se hace para cada ventana y ayuda a  conocer las características en el dominio de la frecuencia de los segmentos ventaneados.

```python
# Función para calcular la mediana en frecuencia
def calcular_mediana_frecuencia(fft_magnitudes, freqs):
    potencia_acumulada = np.cumsum(fft_magnitudes**2)
    potencia_total = potencia_acumulada[-1]
    mediana_idx = np.where(potencia_acumulada >= potencia_total / 2)[0][0]
    return freqs[mediana_idx]

medianas_frecuencia = []
medias_ventanas = []
desviaciones_ventanas = []

for i, segmento in enumerate(ventanas_aventanadas):
    # Calcular la FFT del segmento
    N = len(segmento)
    fft_result = fft(segmento)
    fft_magnitudes = np.abs(fft_result[:N // 2])  # Magnitudes de la FFT (mitad de la señal)
    freqs = fftfreq(N, 1/fs)[:N // 2]  # Frecuencias correspondientes
    
    # Calcular la mediana en frecuencia
    mediana_frecuencia = calcular_mediana_frecuencia(fft_magnitudes, freqs)
    medianas_frecuencia.append(mediana_frecuencia)
    
    # Calcular la media y desviación estándar del segmento ventaneado
    media_segmento = np.mean(segmento)
    desviacion_segmento = np.std(segmento)
    
    medias_ventanas.append(media_segmento)
    desviaciones_ventanas.append(desviacion_segmento)
```
El código, además de aplicar la transformada de Fourier a cada ventana, halla estadísticas descriptivas, media, mediana y desviación estándar de cada uno de las ventanas ya en el dominio de la fecuencia para representar los datos con mayor facilidad, permitiendo el análisis de la distribución de los mismos. 

En cuanto a las medianas, estos fueron los resultados: 

* Medianas en frecuencia de cada ventana:
[44.11764705882352, 35.29411764705882, 30.882352941176467, 57.35294117647058, 48.52941176470588, 52.94117647058823, 44.11764705882352, 57.35294117647058, 35.29411764705882, 61.764705882352935, 35.29411764705882, 30.882352941176467, 30.882352941176467, 39.705882352941174]

Para las medias y desviaciones, en las primeras ventanas se obtuvo lo siguiente: 

* Ventana 1:
Media: -9.370447101183476e-05, 
Desviación estándar: 0.11205817169650448, 
Mediana en frecuencia: 44.11764705882352

* Ventana 2:
Media: -6.810813650019912e-05
Desviación estándar: 0.13351582657994487
Mediana en frecuencia: 35.29411764705882

* Ventana 3:
Media: 0.00011106153308511899
Desviación estándar: 0.24894355321348183
Mediana en frecuencia: 30.882352941176467

* Ventana 4:
Media: 0.0001594586506530195
Desviación estándar: 0.2416175964179292
Mediana en frecuencia: 57.35294117647058

En general, existe una correlación aparente entre la desviación estándar y la mediana en frecuencia. A medida que la mediana en frecuencia disminuye (ventanas 1 a 3), la variabilidad de la señal aumenta. Esto puede interpretarse como una señal más dispersa o menos controlada cuando las frecuencias son más bajas, lo que podría estar relacionado con una mayor relajación muscular o la presencia de ruido.
En la ventana 4, tanto la desviación estándar como la mediana en frecuencia son altas, lo que podría corresponder a un aumento significativo en la actividad muscular o a una contracción intensa.

Adicionalmente, se realizó la gráfica del espectograma y el espectro de cada ventana para evidenciar el comportamiento de los segmentos en el cominio de la frecuencia. A continuación se presentan las gráficas para la primera y última ventana. 

![espectros ventana 1](https://github.com/user-attachments/assets/5e7b7c3d-350c-4b00-86a7-305440fe7d10)

*Figura 10: Espectro y espectograma de la ventana 1. Tomado de : Autoría propia*

El espectro muestra la distribución de las frecuencias a lo largo del tiempo en un formato visual de espectrograma, donde los colores representan la potencia en diferentes bandas de frecuencia. En este caso, se observa que las frecuencias entre 20 Hz y 60 Hz tienen la mayor potencia, indicada por los tonos más claros (verde y amarillo), lo que sugiere que la actividad muscular dominante ocurre en este rango a lo largo del tiempo.

Por otro lado, la señal muestra el espectro de frecuencia resultante de aplicar la Transformada de Fourier a la señal filtrada. Aquí, se identifica un pico principal alrededor de los 45 Hz, lo que indica que esta es la frecuencia dominante de la actividad muscular registrada en este segmento. A diferencia del espectro superior, que varía en función del tiempo, esta gráfica presenta una visión estática que concentra la información en función de la magnitud de cada frecuencia en la señal total.

![espectros ventana 14](https://github.com/user-attachments/assets/227c15e5-8d95-4439-9288-15d60940ad23)

*Figura 11: Espectro y espectograma de la ventana 1. Tomado de : Autoría propia*

Para este segmento se observa que las frecuencias entre los 20 Hz y los 60 Hz son las que tienen mayor potencia. Esto indica que en esa ventana de tiempo, la mayor actividad muscular ocurre dentro de este rango de frecuencias, lo cual es típico en señales EMG. Las frecuencias más altas, por encima de los 100 Hz, tienen menos potencia, lo que sugiere que las frecuencias más relevantes están concentradas en los rangos más bajos.

En la señal, que muestra el espectro de frecuencia tras aplicar la Transformada de Fourier, se observa un pico significativo alrededor de los 45 Hz con una magnitud máxima cercana a 12. Esto indica que esa es la frecuencia dominante de la actividad muscular. También hay otros picos menores entre los 20 Hz y los 100 Hz, lo que sugiere la presencia de otros componentes armónicos, pero las magnitudes caen considerablemente después de los 100 Hz, confirmando que la mayor parte de la actividad relevante ocurre en las frecuencias más bajas.

## Test de hipótesis
El test de hipótesis es un enfoque útil para comparar señales EMG entre diferentes ventanas temporales, como la ventana 1 y la ventana 14, con el fin de determinar si las diferencias observadas son significativas o no. Para realizar este análisis, empleamos el test t de Student, que es una herramienta estadística diseñada para comparar las medias de dos grupos de datos. En este caso, las características clave de la señal EMG, como la magnitud de las frecuencias dominantes o el contenido de potencia, se analizan entre estas dos ventanas. El test t es adecuado para este tipo de datos porque, tras el filtrado, la señal suele tener una distribución aproximadamente normal, y las muestras de cada ventana son relativamente pequeñas.

En particular, el test t de Student es útil cuando se quiere comparar el estado inicial de la señal con el estado al final del proceso , donde pueden aparecer fenómenos como la fatiga muscular. Se podrían observar cambios en la señal, como una disminución en la potencia de las frecuencias altas y una reducción en la magnitud de los picos del espectro, indicadores del fallo muscular. El test t de Student  permite evaluar si estas diferencias son estadísticamente significativas, lo que confirmaría que el fallo ha influido en la señal. De esta forma, este método es una herramienta  para validar cuantitativamente los cambios en las señales EMG a lo largo del tiempo.

Para aplicarlo en el proeyecto se tomó la primer y última ventana, relacionando el test de la siguiente manera. 

* H(0) : La mediana en frecuencia de la primer ventana es igual a la mediana en frecuencia de la última ventana.
* H(1) : La mediana en frecuencia de la primer ventana es menor a la mediana en frecuencia de la última ventana. (La que queremos que sea real).

  
El código es el siguiente  :
  
```python
# %% 
##TEST DE HIPOTESIS

primeras_medianas = medianas_frecuencia[:3]
ultimas_medianas = medianas_frecuencia[-3:]

t_stat, p_value = ttest_rel(primeras_medianas, ultimas_medianas)

if np.mean(primeras_medianas) > np.mean(ultimas_medianas):
  
print(f"Estadístico t: {t_stat}")
print(f"p-valor (unilateral): {p_value}")

# Verificar si se rechaza la hipótesis nula
alpha = 0.05  # Nivel de significancia
if p_value < alpha:
    print("Se rechaza la hipótesis nula: las medianas de las primeras ventanas son significativamente mayores que las últimas.")
else:
    print("No se puede rechazar la hipótesis nula.")
```
Los resultados obtenidos fueron llos siguientes: 

* Estadístico t: 0.45883146774112327
* p-valor: 0.34569665003790817

Teniendo en cuenta el p-valor, no se puede rechazar la hipótesis nula. Por lo tanto, la mediana en frecuencia tanto de la primera como de la última ventana son iguales. A pesar de que al revisar el arreglo de datos de los resultados de las medianas, la primera si resulta mayor que la última, los resultados del test de hipótesis, se deben a que estadísticamente la diferencia no es lo suficientemente relevante para tenerla en cuenta. 

Se buscaba que la hipótesis alternativa fuera real, debido a que se quería demostrar la evidencia de la actividad muscular al llegar a fallo. Al no cumplirlo, se puede deducir que pudo haber incovenientes en la adquisición de la señal y pudieron no haberse guardado toda la cantidad de datos necesarios. 

Adicionalmente, el módulo pudo haber afectado en la correcta adquisición de la señal, distrosionando los resultados reales de la EMG del bíceps. 


[^1^]:Rodríguez, J. (2019). Análisis y procesamiento de señales electromiográficas para la clasificación de actividades de la vida diaria. Universidad Tecnológica de Bolívar.
[^2^]:National Instruments. (s/f). Multifunction Input and Output Devices. https://www.ni.com/pdf/product-flyers/multifunction-io.pdf
[^3^]:ELECTROCARDIOGRAFO ECG AD8232. (s/f). MACTRONICA. https://www.mactronica.com.co/electrocardiografo-ecg-ad8232
[^4^]:Blasco, A. (s/f). ejercicios para bíceps electrodos.  https://www.boteprote.com/blog/ejercicios-para-biceps/ejercicios-para-biceps-electrodos/
[^5^]:Sanz, Á. M. (2017). PROCESADO AVANZADO DE SEÑAL EMG [Carlos III de Madrid, Escuela politécnica superior]. https://e-archivo.uc3m.es/rest/api/core/bitstreams/73de4212-e068-4610-9dca-4cf450e3fd9e/content
[^6^]:González, M. (2017, enero 19). ¿Qué es el «aliasing»? Explicación y efectos curiosos. TECNO CRÓNICA. https://tecnocronica.wordpress.com/2017/01/19/que-es-el-aliasing-explicacion-y-efectos-curiosos/

