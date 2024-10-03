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
* Contracciones = 14

La frecuencia de muestreo fue definida a partir del criterio de nyquist que indica que la fs debe ser más del doble de la frecuencia máxima, en este caso del músculo que estabamos midiendo. De acuerdo con Moreno Sanz, la frecuencia del músculo se encuentra entre 50 – 100 Hz. [^5^]. En este caso particular, se tomó como frecuencia máxima 100 Hz. Se decidió tomar la fs como tres veces la frceuencia máxima, con el fin de evitar el riesgo de aliassing que puede ocurrir si se esta muy cerca del criterio de nyquist. El aliassing es el efecto que tienen las señales continuas en el tiempo de tornarse indistingibles al muestrearse digitalmente. [^6^].

El tiempo de muestreo resultó a partir del desgaste muscular y la tolerancia del sujeto, se logró muestrear apróximadamente 2 minutos. 

En cuanto a las contracciones, debido al ritmo en que se realizó el ejercicio (lento pero continuo), y al corto tiempo muestreado, se lograron apróximadamente 14. 

## Filtrado de la señal
El proceso de filtrado resulta esencial para eliminar las frecuencias no deseadas de la señal y minimizar el ruido cuanto sea posible. Para este caso, se decidió realizar un filtro pasabanda tipo butterworth. El filtro pasabanda permite el paso de un rango específico de frecuencias, lo que resulta esencial en procesamiento de señales EMG, dado que las frecuencias tanto muy altas, como muy bajas resultan ser ruidos e interferencias de los métodos de adquisición. Se eligió como frecuencias de corte 20 y 100 HZ, dado lo contenido en la literaturatura respecto a las frecuencias medias de la actividad muscular. [^5^]. 

Adicionalmente, teniendo en cuenta los parametros de diseño para los filtros, se eligió como frecuencias para atenuación en -20dB, 5 y 120 Hz. Estas frecuencias determinan la pendiente del filtro y el comienzo de la atenuación. 
En cuanto a la elección del tipo butterwoth, se determinó debido a su respuesta en frecuencia plana en la banda pasante, lo que facilita la eliminación de ruidos y artefactos sin introducir ningún tipo de ondulación que pueda afectar la interpretación de los datos. 

Para implementar el filtro en python y aplicarlo a ala señal es necesario conocer el orden del mismo. Este fue determinado de la siguiente manera:

'''''''''''''''''''''FOTO CALCULOS''''''''''''''''''''''''''''''''
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
La ventana que se esta aplicando en cada segmento en el que se detecta un pico, se puede apreciar a continuación:

<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/ventanaaa.png?raw=true"  width="1000" height="500">

*Figura 7: Ventana aplicada. Tomado de : Autoría propia*

En relación a los segmentos elegidos, la gráfica de una de las ventanas comparandola con la contracción sería la siguiente: 


<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/ventana%20y%20segmento1.png?raw=true"  width="1000" height="500">

*Figura 8: Segmento  y ventana de hanning. Tomado de : Autoría propia*

La señal filtrada, con cada segmento ventaneado y la ventana que se le aplica a cada uno se puede observar a continuación: 

<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/se%C3%B1al%20con%20ventanas.png?raw=true"  width="1000" height="500">

*Figura 9: Señal aventanada. Tomado de : Autoría propia*

Por último, se hizo una reconstrucción de la señal a partir de los segmentos aventanados:

<img src="https://github.com/lavaltt/Analisis_de_senales_EMG/blob/main/se%C3%B1al%20reconstruida.png?raw=true"  width="1000" height="500">

*Figura 10: Señal reconstruida. Tomado de : Autoría propia*



## Análisis espectral 
Una vez se cuenta con la señal aventanada, se puede realizar un análisis espectral de cada ventana para conocer las características en este dominio. El análisis espectral se centra en el contenido en frecuencia de la señal, utilizando técnicas como la Transformada de Fourier (FT). 

## Test de hipótesis 


[^1^]:Rodríguez, J. (2019). Análisis y procesamiento de señales electromiográficas para la clasificación de actividades de la vida diaria. Universidad Tecnológica de Bolívar.
[^2^]:National Instruments. (s/f). Multifunction Input and Output Devices. https://www.ni.com/pdf/product-flyers/multifunction-io.pdf
[^3^]:ELECTROCARDIOGRAFO ECG AD8232. (s/f). MACTRONICA. https://www.mactronica.com.co/electrocardiografo-ecg-ad8232
[^4^]:Blasco, A. (s/f). ejercicios para bíceps electrodos.  https://www.boteprote.com/blog/ejercicios-para-biceps/ejercicios-para-biceps-electrodos/
[^5^]:Sanz, Á. M. (2017). PROCESADO AVANZADO DE SEÑAL EMG [Carlos III de Madrid, Escuela politécnica superior]. https://e-archivo.uc3m.es/rest/api/core/bitstreams/73de4212-e068-4610-9dca-4cf450e3fd9e/content
[^6^]:González, M. (2017, enero 19). ¿Qué es el «aliasing»? Explicación y efectos curiosos. TECNO CRÓNICA. https://tecnocronica.wordpress.com/2017/01/19/que-es-el-aliasing-explicacion-y-efectos-curiosos/

