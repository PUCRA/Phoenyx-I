#!/usr/bin/env python3
import time
import math
from dataclasses import dataclass

import board
import busio
import adafruit_mpu6050


def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_from_euler(roll: float, pitch: float, yaw: float):
    """Quaternion (x,y,z,w) from roll,pitch,yaw."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


@dataclass
class ImuState:
    roll: float = 0.0   # rad
    pitch: float = 0.0  # rad
    yaw: float = 0.0    # rad
    gyro_bias_z: float = 0.0  # rad/s


class MPU6050Fusion:
    """
    Fusión 6DOF "profesional" para MPU-6050 sin magnetómetro:
    - Roll/pitch: complementario (acel corrige deriva)
    - Yaw: integración de gyro Z con:
        * calibración inicial
        * estimación adaptativa de bias cuando está quieto (ZUPT)
    Resultado: buen heading a corto/medio plazo, deriva muy reducida.
    """
    def __init__(
        self,
        address=0x68,
        rate_hz=100.0,
        alpha_rp=0.98,               # complementario roll/pitch (0.95-0.99 típico)
        still_accel_tol=0.15,        # m/s^2 tolerancia respecto a |g|
        still_gyro_tol=0.10,         # rad/s tolerancia en norma gyro
        bias_learn_rate=0.02,        # cuánto ajusta bias por actualización (0.01-0.05)
        init_bias_seconds=2.0        # tiempo de calibración inicial en reposo
    ):
        self.dt_target = 1.0 / rate_hz
        self.alpha_rp = alpha_rp
        self.still_accel_tol = still_accel_tol
        self.still_gyro_tol = still_gyro_tol
        self.bias_learn_rate = bias_learn_rate
        self.init_bias_seconds = init_bias_seconds

        i2c = busio.I2C(board.SCL, board.SDA)
        self.mpu = adafruit_mpu6050.MPU6050(i2c, address=address)

        # (Opcional) puedes ajustar rangos si quieres:
        # self.mpu.gyro_range = adafruit_mpu6050.GyroRange.RANGE_250_DPS
        # self.mpu.accelerometer_range = adafruit_mpu6050.Range.RANGE_2_G

        self.state = ImuState()
        self._last_t = time.monotonic()

        self._calibrate_initial_bias()

    def _calibrate_initial_bias(self):
        """Calibra bias de gyro Z en reposo."""
        t0 = time.monotonic()
        acc = 0.0
        n = 0
        print(f"[IMU] Calibrando bias gyro Z durante {self.init_bias_seconds:.1f}s... NO MOVER")
        while time.monotonic() - t0 < self.init_bias_seconds:
            gx, gy, gz = self.mpu.gyro  # rad/s
            acc += gz
            n += 1
            time.sleep(0.005)
        self.state.gyro_bias_z = acc / max(1, n)
        print(f"[IMU] Bias inicial gyro Z = {self.state.gyro_bias_z:.6f} rad/s (n={n})")

        # Inicializa roll/pitch desde accel (más estable)
        ax, ay, az = self.mpu.acceleration  # m/s^2
        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        self.state.roll = roll_acc
        self.state.pitch = pitch_acc
        self.state.yaw = 0.0

        self._last_t = time.monotonic()

    def _is_still(self, ax, ay, az, gx, gy, gz) -> bool:
        g = 9.80665
        acc_norm = math.sqrt(ax*ax + ay*ay + az*az)
        gyro_norm = math.sqrt(gx*gx + gy*gy + gz*gz)

        accel_ok = abs(acc_norm - g) < self.still_accel_tol
        gyro_ok = gyro_norm < self.still_gyro_tol
        return accel_ok and gyro_ok

    def update(self):
        """Devuelve (state, accel(m/s^2), gyro(rad/s), quat(x,y,z,w), dt)."""
        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now

        # Lectura
        ax, ay, az = self.mpu.acceleration  # m/s^2
        gx, gy, gz = self.mpu.gyro          # rad/s

        # 1) Roll/Pitch desde accel (referencia gravedad)
        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        # 2) Roll/Pitch gyro integración + complementario
        roll_gyro = self.state.roll + gx * dt
        pitch_gyro = self.state.pitch + gy * dt

        self.state.roll = self.alpha_rp * roll_gyro + (1.0 - self.alpha_rp) * roll_acc
        self.state.pitch = self.alpha_rp * pitch_gyro + (1.0 - self.alpha_rp) * pitch_acc

        # 3) Bias adaptativo en yaw cuando está quieto (ZUPT)
        if self._is_still(ax, ay, az, gx, gy, gz):
            # Si está quieto, “lo correcto” para gz es ~0 (sin giro real).
            # Ajustamos bias para que (gz - bias) -> 0.
            err = gz - self.state.gyro_bias_z
            self.state.gyro_bias_z += self.bias_learn_rate * err

        # 4) Yaw por integración de gz - bias
        gz_unbiased = gz - self.state.gyro_bias_z
        self.state.yaw = wrap_pi(self.state.yaw + gz_unbiased * dt)

        q = quat_from_euler(self.state.roll, self.state.pitch, self.state.yaw)

        # Mantener un ritmo aproximado si lo usas en bucle
        return self.state, (ax, ay, az), (gx, gy, gz_unbiased), q, dt


def main():
    fusion = MPU6050Fusion(
        address=0x68,
        rate_hz=100.0,
        alpha_rp=0.98,
        still_accel_tol=0.20,   # ajusta si vibra mucho
        still_gyro_tol=0.12,    # ajusta si vibra mucho
        bias_learn_rate=0.02,
        init_bias_seconds=2.0
    )

    print("Formato: roll pitch yaw [deg] | bias_z [rad/s] | gz_unbiased [deg/s]")
    try:
        while True:
            st, acc, gyro, q, dt = fusion.update()
            gx, gy, gz_u = gyro

            print(
                f"RPY: {math.degrees(st.roll):+7.2f} {math.degrees(st.pitch):+7.2f} {math.degrees(st.yaw):+7.2f} | "
                f"bias_z: {st.gyro_bias_z:+.6f} | "
                f"gz_u: {math.degrees(gz_u):+7.3f}"
            )

            # bucle ~100 Hz
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()