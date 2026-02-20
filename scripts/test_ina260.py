#!/usr/bin/env python3
import time
import board
import busio
from adafruit_ina260 import INA260


def main():
    print("[INA260] Inicializando bus I2C...")
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        ina = INA260(i2c, address=0x45)
    except Exception as e:
        print("[ERROR] No se pudo inicializar INA260")
        print(e)
        return

    print("[INA260] Sensor detectado correctamente")
    print("Formato: Voltaje [V] | Corriente [mA] | Potencia [mW]")
    print("----------------------------------------------------")

    last_print = time.time()

    try:
        while True:
            voltage = ina.voltage        # V
            current = ina.current        # mA
            power = ina.power            # mW

            # Filtro bÃ¡sico de sanity-check
            if voltage < 0 or voltage > 60:
                print("[WARN] Voltaje fuera de rango:", voltage)

            if abs(current) > 30000:
                print("[WARN] Corriente sospechosa:", current)

            now = time.time()
            if now - last_print >= 0.5:  # 2 Hz
                print(
                    f"V={voltage:6.2f} V | "
                    f"I={current:8.2f} mA | "
                    f"P={power:8.2f} mW"
                )
                last_print = now

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INA260] Test detenido por usuario")

    except Exception as e:
        print("\n[ERROR] Fallo leyendo INA260")
        print(e)


if __name__ == "__main__":
    main()