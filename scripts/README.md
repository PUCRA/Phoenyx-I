# 📡 Unitree LIDAR L2 4D – Automatic Network Setup Script

Bash script to automatically configure your PC’s Ethernet interface to communicate with the **Unitree LIDAR L2 4D**.

The script detects the correct network interface, assigns a compatible static IP address, and verifies connectivity with the LIDAR.

---

## 🚀 Features

- 🔍 Automatic Ethernet interface detection  
- 👤 Interactive selection if multiple interfaces are found  
- 🔐 Automatic sudo permission handling  
- 🌐 Static IP configuration  
- ✅ Connectivity verification using `ping`  
- 💾 Optional permanent configuration (`--permanent`)  
  - NetPlan support  
  - NetworkManager support  

---

## 🌐 Network Configuration

| Device | IP Address | Subnet |
|--------|------------|--------|
| PC | `192.168.1.100` | `/24` |
| LIDAR | `192.168.1.62` | `/24` |
| Network | `192.168.1.0/24` | — |

---

## 📋 Requirements

- Linux system (Ubuntu recommended)  
- `sudo` privileges  
- Ethernet cable  
- LIDAR powered on  

---

## 🛠 Installation

Place the script inside your project directory.

Grant execution permissions:

```bash
chmod +x setup_lidar_network.sh
```

## ▶️ Usage
Temporary Configuration (Default)
```bash
./setup_network_lidar.sh
```

This will:
- Detect available Ethernet interfaces
- Allow you to select one (if multiple are found)
- Assign static IP 192.168.1.100/24
- Test connectivity with 192.168.1.62

## Expected Output
```bash

==========================================
  CONFIGURACIÓN RED UNITREE LIDAR L2 4D
==========================================

[1/5] Detectando interfaces Ethernet...

Interfaces de red disponibles:
  - lo
  - eno1
  - wlo1

✅ Usando interfaz: eno1

[2/5] Verificando permisos...
⚠️  Este script necesita permisos sudo
Por favor, introduce tu contraseña:
[sudo] password for judith: 
✅ Permisos verificados

[3/5] Configurando interfaz eno1...

  → Desactivando NetworkManager para eno1...
  → Bajando interfaz...
  → Limpiando configuración IP anterior...
  → Levantando interfaz...
  → Asignando IP 192.168.1.100/24...
✅ Interfaz configurada

[4/5] Verificando configuración...

  Configuración actual de eno1:
  IP: 192.168.1.100/24
  Estado: DOWN

✅ Configuración verificada

[5/5] Probando conexión con el LIDAR (192.168.1.62)...

```