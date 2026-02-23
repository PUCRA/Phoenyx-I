import serial
import time

PORT = "/dev/ttyACM0"
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

print("=== ROVER LIGHT CONTROL ===")
print("Formato: EFECTOCOLOR")
print("Ejemplo: 90")
print("Escribe 'exit' para salir\n")

while True:
    cmd = input(">> ")

    if cmd.lower() == "exit":
        break

    if len(cmd) != 2 or not cmd.isdigit():
        print("Comando inválido. Usa 2 números.")
        continue

    ser.write((cmd + "\n").encode())
    print(f"Enviado: {cmd}")

ser.close()