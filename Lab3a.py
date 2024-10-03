# -- coding: utf-8 --
"""
Created on Wed Sep 25 22:56:48 2024

@author: Automatizacion 2
"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import numpy as np
from scipy.signal import windows
from scipy.fft import fft, fftfreq
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.signal import spectrogram

# Cargar los datos
datos = pd.read_csv(r'C:\Users\Automatizacion 2\Downloads\V\Señales\lab_señales\voltajes_señal3.csv')
EMG = datos['Voltaje (V)'] 

# Convertir la columna a tipo numérico
datos['Voltaje (V)'] = pd.to_numeric(datos['Voltaje (V)'])

# Graficar la señal original
plt.figure(figsize=(12,6))
plt.plot(datos['Voltaje (V)'],color='red')
plt.xlabel('Tiempo (ms)')
plt.ylabel ('Amplitud (V)')
plt.title('Señal EMG adquirida del bíceps')
plt.grid()
plt.show()

#%% FILTRADO DE LA SEÑAL
fs = 300.0  # frecuencia de muestreo

lc = 20.0   # frecuencia de corte baja
hc = 100.0  # frecuencia de corte alta
n = 5       # orden del filtro

# Normalizar las frecuencias de corte
nyquist = 0.5 * fs
low = lc / nyquist
high = hc / nyquist

# Coeficientes del filtro Butterworth
b, a = signal.butter(n, [low, high], btype='band')

# Filtrar la señal
datos_filtrados = signal.filtfilt(b, a, datos['Voltaje (V)'])

# Graficar la señal original y la señal filtrada
plt.figure(figsize=(12, 6))

# Señal original
plt.subplot(2, 1, 1)
plt.plot(datos['Voltaje (V)'], color='red')
plt.title('Señal Original adquirida')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud [V]')
plt.grid()

# Señal filtrada
plt.subplot(2, 1, 2)
plt.plot(datos_filtrados, color='blue')
plt.title('Señal Filtrada - Butterworth Pasabanda')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud [V]')
plt.grid()

plt.tight_layout()
plt.show()

#%% AVENTANAMIENTO DE LA SEÑAL
# Detectar contracciones (picos en la señal EMG filtrada)
threshold = 0.005  # Ajusta este valor según tus datos
picos, _ = signal.find_peaks(datos_filtrados, height=threshold, distance=fs * 0.2)  # Mínimo 0.5s entre picos

# Definir las ventanas
ventana_size = int(0.115 * fs)  # 200 ms antes y después de cada contracción
ventana_total = 2 * ventana_size

# Parámetro para variar las ventanas
amplitud_variacion = 0.0005  # Ajusta este valor para controlar la magnitud de la variación

# Crear listas para almacenar los segmentos ventaneados y las ventanas de Hanning
ventanas_aventanadas = []
ventanas_hanning = []
tiempos_ventanas = []

# Aplicar la ventana de Hanning con variación a cada segmento
for i, pico in enumerate(picos):
    inicio = pico - ventana_size
    fin = pico + ventana_size
    
    # Asegúrate de que no se salga del rango de la señal
    if inicio < 0 or fin > len(datos_filtrados):
        continue
    
    # Crear la ventana de Hanning
    ventana_hanning = windows.hann(ventana_total)
    
    # Extraer el segmento y aplicar la ventana
    segmento = datos_filtrados[inicio:fin] * ventana_hanning
    
    # Agregar una pequeña variación aleatoria a cada ventana
    variacion = np.random.normal(0, amplitud_variacion, len(segmento))
    segmento_variado = segmento + variacion
    
    # Guardar el segmento variado y sus tiempos
    ventanas_aventanadas.append(segmento_variado)
    ventanas_hanning.append(ventana_hanning)
    tiempos_ventanas.append(np.arange(inicio, fin) / fs)
    
    # Graficar la ventana de Hanning y el segmento ventaneado para cada ventana
    plt.figure(figsize=(10, 4))
    
    # Ventana de Hanning
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(ventana_total) / fs, ventana_hanning, color='green')
    plt.title(f'Ventana de Hanning {i + 1}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid()
    
    # Segmento ventaneado
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(ventana_total) / fs, segmento_variado, color='blue')
    plt.title(f'Segmento Ventaneado {i + 1}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid()
    
    plt.tight_layout()
    plt.show()

# Graficar la ventana de Hanning aplicada
plt.figure(figsize=(12, 6))
plt.plot(np.arange(ventana_total) / fs, ventana_hanning, color='green', label='Ventana de Hanning')
plt.title('Ventana de Hanning Aplicada')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend(fontsize=8)
plt.grid()
plt.xlim(0, ventana_total / fs)
plt.show()

# Graficar la señal filtrada y las ventanas con variación
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(datos_filtrados)) / fs, datos_filtrados, color='pink', label='Señal Filtrada')

# Graficar cada ventana y la ventana de Hanning
for i, (ventana, ventana_hanning) in enumerate(zip(ventanas_aventanadas, ventanas_hanning)):
    tiempo = tiempos_ventanas[i]
    
    # Graficar la ventana de Hanning
    plt.plot(tiempo, ventana_hanning * np.max(ventana), color='black', linestyle='--', label=f'Ventana Hanning' if i == 0 else "")
    
    # Graficar el segmento ventaneado
    plt.plot(tiempo, ventana, label=f'Ventana {i + 1}', alpha=0.75)

plt.title('Señal EMG filtrada y ventanas de Hanning aplicadas')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend(fontsize=6)
plt.grid()
plt.show()

# Crear un vector para la señal reconstruida con variación
signal_reconstructed_variada = np.zeros(len(datos_filtrados))

# Aplicar la ventana de Hanning con variación a cada segmento y reconstruir la señal
for pico in picos:
    inicio = pico - ventana_size
    fin = pico + ventana_size
    
    # Asegúrate de que no se salga del rango de la señal
    if inicio < 0 or fin > len(datos_filtrados):
        continue
    
    # Crear la ventana de Hanning
    ventana_hanning = windows.hann(ventana_total)
    
    # Extraer el segmento y aplicar la ventana
    segmento = datos_filtrados[inicio:fin] * ventana_hanning
    
    # Agregar una pequeña variación aleatoria a cada ventana
    variacion = np.random.normal(0, amplitud_variacion, len(segmento))
    segmento_variado = segmento + variacion
    
    # Sumar el segmento ventaneado variado a la señal reconstruida
    signal_reconstructed_variada[inicio:fin] += segmento_variado  # Sumar en la posición correspondiente

# Graficar la señal reconstruida con variación
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(signal_reconstructed_variada)) / fs, signal_reconstructed_variada, color='magenta', label='Señal Reconstruida')
plt.title('Señal EMG reconstruida a partir de segmentos ventaneados')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend(fontsize=8)
plt.grid()
plt.show()

#%% ANÁLISIS ESPECTRAL DE CADA VENTANA USANDO FFT
# Función para calcular la mediana en frecuencia
def calcular_mediana_frecuencia(fft_magnitudes, freqs):
    potencia_acumulada = np.cumsum(fft_magnitudes**2)
    potencia_total = potencia_acumulada[-1]
    mediana_idx = np.where(potencia_acumulada >= potencia_total / 2)[0][0]
    return freqs[mediana_idx]

# Inicializar listas para guardar resultados
medianas_frecuencia = []
medias_ventanas = []
desviaciones_ventanas = []

# Calcular el espectrograma para la señal completa antes del bucle
f_spectro, t_spectro, Sxx = spectrogram(datos_filtrados, fs)

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
    
    # Almacenar la media y desviación estándar
    medias_ventanas.append(media_segmento)
    desviaciones_ventanas.append(desviacion_segmento)
    
    # Graficar el espectrograma y el espectro de la ventana en una sola figura
    plt.figure(figsize=(12, 6))

    # Subplot 1: Espectrograma
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t_spectro, f_spectro, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Potencia/Frecuencia [dB/Hz]')
    plt.ylabel('Frecuencia [Hz]')
    plt.title(f'Espectrograma y espectro de la ventana {i+1}')
    
    # Subplot 2: Espectro de la ventana
    plt.subplot(2, 1, 2)
    plt.plot(freqs, fft_magnitudes, color='blue')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.grid()

    # Mostrar la figura
    plt.tight_layout()
    plt.show()

    # Mostrar la media, desviación estándar y mediana en frecuencia de la ventana actual
    print(f"\nVentana {i+1}:")
    print(f"Media: {media_segmento}")
    print(f"Desviación estándar: {desviacion_segmento}")
    print(f"Mediana en frecuencia: {mediana_frecuencia}")

# Mostrar las medianas en frecuencia de todas las ventanas
print("Medianas en frecuencia de cada ventana:")
print(medianas_frecuencia)

# %% 
##TEST DE HIPOTESIS
# Tomar las primeras y últimas 3 medianas para hacer una comparación más significativa
primeras_medianas = medianas_frecuencia[:3]
ultimas_medianas = medianas_frecuencia[-3:]

# Realizar la prueba t para muestras relacionadas (bilateral) entre las primeras y últimas medianas
t_stat, p_value = ttest_rel(primeras_medianas, ultimas_medianas)

# Como la hipótesis alternativa es que las primeras son mayores, ajustamos el p-valor a una prueba unilateral
if np.mean(primeras_medianas) > np.mean(ultimas_medianas):
    p_value /= 2  # Dividir el p-valor por 2 para la prueba unilateral

# Mostrar los resultados
print(f"Estadístico t: {t_stat}")
print(f"p-valor (unilateral): {p_value}")

# Verificar si se rechaza la hipótesis nula
alpha = 0.05  # Nivel de significancia
if p_value < alpha:
    print("Se rechaza la hipótesis nula: las medianas de las primeras ventanas son significativamente mayores que las últimas.")
else:
    print("No se puede rechazar la hipótesis nula.")