import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import os

# Crear una carpeta donde se guardarán los datos (ruta personalizada)
output_folder = 'C:/Users/Automatizacion 2/Downloads/V/lab_señales'
os.makedirs(output_folder, exist_ok=True)

# Nombre del archivo dentro de la carpeta
output_file = os.path.join(output_folder, 'voltajes_señal3.csv')  # Cambié el nombre del archivo

# Parámetros de configuración
device_name = 'Dev1'  # Cambia esto al nombre de tu dispositivo DAQ si es diferente
analog_input_channel = f'{device_name}/ai0'
sample_rate = 10  # Frecuencia de muestreo (muestras por segundo)
num_samples = 1  # Número de muestras por lectura
duration = 120  # Duración total de adquisición en segundos

# Variables para guardar los datos en lotes
batch_size = 10  # Número de muestras a guardar en cada lote
data_buffer = []  # Buffer para almacenar los datos

# Configurar la tarea de adquisición
with nidaqmx.Task() as task:
    # Agregar un canal de entrada analógica
    task.ai_channels.add_ai_voltage_chan(analog_input_channel)
    
    # Configurar la adquisición continua
    task.timing.cfg_samp_clk_timing(rate=sample_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
    
    # Inicializar variables para la gráfica en tiempo real
    plt.ion()  # Habilitar modo interactivo
    fig, ax = plt.subplots()
    total_samples = int(sample_rate * duration)  # Total de muestras a adquirir
    xdata = np.arange(0, total_samples)  # Eje X
    ydata = np.zeros(total_samples)  # Inicializa los datos Y
    line, = ax.plot(xdata, ydata)
    ax.set_ylim(0, 3.5)  # Ajusta los límites del eje Y según tu señal
    ax.set_xlim(0, total_samples)  # Limitar el eje X a la cantidad total de muestras
    ax.set_xlabel('Muestras')
    ax.set_ylabel('Voltaje (V)')
    ax.set_title('Señal en Tiempo Real')

    # Abrir el archivo CSV para guardar los voltajes
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Voltaje (V)"])  # Encabezado del archivo CSV

        # Iniciar la adquisición
        start_time = time.time()
        sample_count = 0  # Contador de muestras

        while time.time() - start_time < duration:
            # Leer la muestra
            data = task.read(number_of_samples_per_channel=num_samples)
            voltage = data[0]  # Extraer el voltaje
            ydata[sample_count] = voltage  # Almacenar el voltaje en el array
            data_buffer.append(voltage)  # Agregar voltaje al buffer

            # Si el buffer alcanza el tamaño del lote, escribir en el archivo CSV
            if len(data_buffer) >= batch_size:
                writer.writerows([[v] for v in data_buffer])  # Guardar el lote
                data_buffer = []  # Limpiar el buffer

            # Actualizar la gráfica
            line.set_ydata(ydata)  # Actualizar la gráfica
            fig.canvas.draw()  # Redibujar la figura
            fig.canvas.flush_events()  # Procesar eventos pendientes
            plt.pause(0.01)  # Pequeña pausa para permitir que la gráfica se actualice
            
            sample_count += 1  # Incrementar el contador de muestras

        # Guardar cualquier dato restante en el buffer al final
        if data_buffer:
            writer.writerows([[v] for v in data_buffer])

    print(f"Datos de voltaje guardados en: {output_file}")

# Finalizar la gráfica
plt.ioff()  # Desactivar modo interactivo
plt.show()  # Mostrar la gráfica final fuera del bucle
