#!/bin/bash

#############################################################################
# CONFIGURACIÓN AUTOMÁTICA DE RED PARA UNITREE LIDAR L2 4D
#############################################################################
# Este script detecta automáticamente tu interfaz Ethernet y la configura
# para comunicarse con el LIDAR (IP: 192.168.1.62)
#
# Uso:
#   chmod +x setup_lidar_network.sh
#   ./setup_lidar_network.sh
#############################################################################

set -e  # Salir si hay algún error

echo ""
echo "=========================================="
echo "  CONFIGURACIÓN RED UNITREE LIDAR L2 4D"
echo "=========================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#############################################################################
# PASO 1: DETECTAR INTERFAZ ETHERNET
#############################################################################

echo -e "${BLUE}[1/5] Detectando interfaces Ethernet...${NC}"
echo ""

# Listar todas las interfaces de red
echo "Interfaces de red disponibles:"
ip link show | grep -E "^[0-9]+:" | awk '{print "  - " $2}' | sed 's/://g'
echo ""

# Detectar interfaces Ethernet (excluir loopback, wireless, virtuales)
ETHERNET_INTERFACES=$(ip link show | grep -E "^[0-9]+:" | awk '{print $2}' | sed 's/://g' | grep -vE "^(lo|wl|docker|br-|veth|virbr)")

# Contar cuántas interfaces Ethernet hay
INTERFACE_COUNT=$(echo "$ETHERNET_INTERFACES" | wc -l)

if [ -z "$ETHERNET_INTERFACES" ]; then
    echo -e "${RED}❌ No se detectaron interfaces Ethernet${NC}"
    echo ""
    echo "Interfaces disponibles:"
    ip link show
    exit 1
fi

# Si hay múltiples interfaces, dejar que el usuario elija
if [ $INTERFACE_COUNT -gt 1 ]; then
    echo -e "${YELLOW}Se detectaron múltiples interfaces Ethernet:${NC}"
    echo ""
    
    # Mostrar interfaces con más detalles
    i=1
    declare -a IFACE_ARRAY
    while IFS= read -r iface; do
        IFACE_ARRAY[$i]=$iface
        
        # Obtener estado
        STATE=$(ip link show $iface | grep -oP 'state \K\w+')
        
        # Obtener MAC
        MAC=$(ip link show $iface | grep -oP 'link/ether \K[a-f0-9:]+')
        
        echo "  [$i] $iface"
        echo "      Estado: $STATE"
        echo "      MAC: $MAC"
        echo ""
        
        ((i++))
    done <<< "$ETHERNET_INTERFACES"
    
    echo -n "Selecciona la interfaz conectada al LIDAR [1-$INTERFACE_COUNT]: "
    read SELECTION
    
    if [ "$SELECTION" -lt 1 ] || [ "$SELECTION" -gt $INTERFACE_COUNT ]; then
        echo -e "${RED}❌ Selección inválida${NC}"
        exit 1
    fi
    
    INTERFACE=${IFACE_ARRAY[$SELECTION]}
else
    # Solo hay una interfaz Ethernet
    INTERFACE=$ETHERNET_INTERFACES
fi

echo -e "${GREEN}✅ Usando interfaz: $INTERFACE${NC}"
echo ""

#############################################################################
# PASO 2: VERIFICAR PERMISOS SUDO
#############################################################################

echo -e "${BLUE}[2/5] Verificando permisos...${NC}"

if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  Este script necesita permisos sudo${NC}"
    echo "Por favor, introduce tu contraseña:"
    sudo -v
fi

echo -e "${GREEN}✅ Permisos verificados${NC}"
echo ""

#############################################################################
# PASO 3: CONFIGURAR INTERFAZ
#############################################################################

echo -e "${BLUE}[3/5] Configurando interfaz $INTERFACE...${NC}"
echo ""

# Quitar gestión de NetworkManager
echo "  → Desactivando NetworkManager para $INTERFACE..."
sudo nmcli device set $INTERFACE managed no 2>/dev/null || true

# Bajar interfaz
echo "  → Bajando interfaz..."
sudo ip link set $INTERFACE down

# Limpiar configuración IP anterior
echo "  → Limpiando configuración IP anterior..."
sudo ip addr flush dev $INTERFACE

# Levantar interfaz
echo "  → Levantando interfaz..."
sudo ip link set $INTERFACE up

# Asignar IP estática
echo "  → Asignando IP 192.168.1.100/24..."
sudo ip addr add 192.168.1.100/24 dev $INTERFACE

echo -e "${GREEN}✅ Interfaz configurada${NC}"
echo ""

#############################################################################
# PASO 4: VERIFICAR CONFIGURACIÓN
#############################################################################

echo -e "${BLUE}[4/5] Verificando configuración...${NC}"
echo ""

# Mostrar configuración IP
IP_CONFIG=$(ip addr show $INTERFACE | grep "inet " | awk '{print $2}')

if [ -z "$IP_CONFIG" ]; then
    echo -e "${RED}❌ Error: No se pudo asignar la IP${NC}"
    exit 1
fi

echo "  Configuración actual de $INTERFACE:"
echo "  IP: $IP_CONFIG"
echo "  Estado: $(ip link show $INTERFACE | grep -oP 'state \K\w+')"
echo ""

echo -e "${GREEN}✅ Configuración verificada${NC}"
echo ""

#############################################################################
# PASO 5: PROBAR CONEXIÓN CON EL LIDAR
#############################################################################

echo -e "${BLUE}[5/5] Probando conexión con el LIDAR (192.168.1.62)...${NC}"
echo ""

# Esperar un poco para que la interfaz se estabilice
sleep 2

# Hacer ping al LIDAR
echo "Enviando 3 pings al LIDAR..."
if ping -c 3 -W 2 192.168.1.62 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ ¡CONEXIÓN EXITOSA CON EL LIDAR!${NC}"
    echo ""
    ping -c 3 192.168.1.62 | tail -3
else
    echo -e "${RED}❌ No se pudo conectar con el LIDAR${NC}"
    echo ""
    echo "Posibles causas:"
    echo "  1. El LIDAR no está conectado físicamente"
    echo "  2. El LIDAR no está encendido"
    echo "  3. El cable Ethernet está dañado"
    echo "  4. El LIDAR tiene una IP diferente"
    echo ""
    echo "Verifica la conexión física y vuelve a intentar."
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}  ✅ CONFIGURACIÓN COMPLETADA${NC}"
echo "=========================================="
echo ""
echo "Tu PC está lista para usar el LIDAR Unitree L2 4D"
echo ""
echo "Información de la configuración:"
echo "  • Interfaz: $INTERFACE"
echo "  • IP del PC: 192.168.1.100"
echo "  • IP del LIDAR: 192.168.1.62"
echo "  • Red: 192.168.1.0/24"
echo ""
echo -e "${YELLOW}IMPORTANTE:${NC}"
echo "Esta configuración es temporal y se perderá al reiniciar."
echo ""

#############################################################################
# OPCIÓN: CONFIGURACIÓN PERMANENTE (NetPlan o NetworkManager)
#############################################################################

if [ "$1" == "--permanent" ]; then
    echo "=========================================="
    echo "  CONFIGURACIÓN PERMANENTE"
    echo "=========================================="
    echo ""
    
    # Detectar si usa NetPlan o NetworkManager
    if [ -d "/etc/netplan" ]; then
        echo "Sistema usa NetPlan. Creando configuración..."
        
        NETPLAN_FILE="/etc/netplan/99-lidar-static.yaml"
        
        sudo tee $NETPLAN_FILE > /dev/null <<EOF
# Configuración estática para LIDAR Unitree L2 4D
network:
  version: 2
  renderer: networkd
  ethernets:
    $INTERFACE:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
EOF
        
        echo "Aplicando configuración NetPlan..."
        sudo netplan apply
        
        echo -e "${GREEN}✅ Configuración permanente creada en $NETPLAN_FILE${NC}"
        
    else
        echo "Sistema usa NetworkManager. Creando conexión..."
        
        # Eliminar conexión anterior si existe
        sudo nmcli connection delete "LIDAR-Static" 2>/dev/null || true
        
        # Crear nueva conexión estática
        sudo nmcli connection add \
            type ethernet \
            con-name "LIDAR-Static" \
            ifname $INTERFACE \
            ipv4.method manual \
            ipv4.addresses 192.168.1.100/24
        
        # Activar conexión
        sudo nmcli connection up "LIDAR-Static"
        
        echo -e "${GREEN}✅ Configuración permanente creada: LIDAR-Static${NC}"
        echo ""
        echo "Para activarla manualmente:"
        echo "  sudo nmcli connection up LIDAR-Static"
        echo ""
        echo "Para volver a DHCP:"
        echo "  sudo nmcli connection down LIDAR-Static"
    fi
    
    echo ""
fi

echo "¡Listo para usar el LIDAR! 🚀"
echo ""