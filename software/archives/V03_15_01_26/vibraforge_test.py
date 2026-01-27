#main code for testing the Vibraforge device
import asyncio
import argparse
from bleak import BleakScanner, BleakClient


CHARACTERISTIC_UUID = 'f22535de-5375-44bd-8ca9-d0ea9ff9e410'
CONTROL_UNIT_NAME = 'QT Py ESP32-S3'


'''
command format
First byte: 00XXXX0Y, X is serial group number, Y is mode,
Second byte: 01XXXXXX, X is address,
Third byte: 1XXXXYYZ, X is duty, Y is frequency, Z is wave.
'''


def create_command(addr, mode, duty, freq):
    serial_group = addr // 16
    serial_addr = addr % 16
    byte1 = (serial_group << 2) | (mode & 0x01)
    byte2 = 0x40 | (serial_addr & 0x3F)  # 0x40 represents the leading '01'
    byte3 = 0x80 | ((duty & 0x0F) << 3) | (freq)  # 0x80 represents the leading '1'
    return bytearray([byte1, byte2, byte3])


async def send_stop_signal(client, addr, freq):
    """Envoie un signal d'arrêt (mode 0) pour couper la fréquence"""
    command = bytearray([])
    command = command + create_command(addr, 0, 0, freq)  # mode=0 pour arrêter
    command = command + bytearray([0xFF, 0xFF, 0xFF]) * 19  # Padding
    await client.write_gatt_char(CHARACTERISTIC_UUID, command)


async def send_vibration_command(client, addr, duty, freq, duration=1.0):
    """Envoie une commande de vibration et attend avant d'arrêter"""
    command = bytearray([])
    command = command + create_command(addr, 1, duty, freq)  # mode=1 pour démarrer
    command = command + bytearray([0xFF, 0xFF, 0xFF]) * 19  # Padding
    await client.write_gatt_char(CHARACTERISTIC_UUID, command)
    await asyncio.sleep(duration)
    await send_stop_signal(client, addr, freq)
    await asyncio.sleep(0.5)  # Délai de refroidissement entre les tests


async def test_single_address(client, addr):
    """Teste une adresse spécifique avec tous les seuils de puissance"""
    print(f"\n{'='*60}")
    print(f"Test de l'adresse {addr}")
    print(f"{'='*60}")
    
    freq = int(input(f'Fréquence à utiliser (0-7) pour l\'adresse {addr}? '))
    
    for duty in range(16):  # 0-15 pour les seuils de puissance
        print(f"\nTest: Adresse={addr}, Puissance={duty}/15, Fréquence={freq}")
        print("Vibration en cours... (1 seconde)")
        await send_vibration_command(client, addr, duty, freq, duration=1.0)
        
        continue_test = input("Continuer au seuil suivant? (o/n): ").strip().lower()
        if continue_test != 'o':
            break
    
    print(f"Test de l'adresse {addr} terminé\n")


async def test_all_addresses(client):
    """Teste automatiquement toutes les adresses avec tous les seuils"""
    print(f"\n{'='*60}")
    print("MODE: Test complet de toutes les adresses")
    print(f"{'='*60}")
    
    start_addr = int(input("Adresse de départ (0-127)? "))
    end_addr = int(input("Adresse de fin (0-127)? "))
    freq = int(input("Fréquence à utiliser (0-7)? "))
    duration = float(input("Durée de vibration par test (secondes)? "))
    
    for addr in range(start_addr, end_addr + 1):
        print(f"\n--- Adresse {addr} ---")
        for duty in range(16):
            print(f"Adresse={addr}, Puissance={duty}/15, Fréquence={freq}")
            await send_vibration_command(client, addr, duty, freq, duration=duration)
        
        pause = input(f"Pause après adresse {addr}? (o/n): ").strip().lower()
        if pause == 'o':
            pause_time = float(input("Durée de la pause (secondes)? "))
            await asyncio.sleep(pause_time)
    
    print("\nTest complet terminé\n")


async def test_manual_mode(client):
    """Mode manuel pour tester des commandes individuelles"""
    while True:
        motor_addr = int(input('Adresse du moteur à contrôler (0-127)? '))
        duty = int(input('Puissance (0-15)? '))
        freq = int(input('Fréquence (0-7)? '))
        start_or_stop = int(input('1 pour démarrer, 0 pour arrêter? '))
        
        user_input = {
            'addr': motor_addr,
            'mode': start_or_stop,
            'duty': duty,
            'freq': freq
        }
        
        command = bytearray([])
        command = command + create_command(user_input['addr'], user_input['mode'], user_input['duty'], user_input['freq'])
        command = command + bytearray([0xFF, 0xFF, 0xFF]) * 19  # Padding
        await client.write_gatt_char(CHARACTERISTIC_UUID, command)
        
        if start_or_stop == 1:
            print("Vibration envoyée. Attente de 2 secondes avant arrêt...")
            await asyncio.sleep(2)
            await send_stop_signal(client, user_input['addr'], user_input['freq'])
            print("Signal d'arrêt envoyé")
        
        another = input("\nTester une autre commande? (o/n): ").strip().lower()
        if another != 'o':
            break


async def setMotor(client):
    """Menu principal pour choisir le mode de test"""
    while True:
        print(f"\n{'='*60}")
        print("MENU DE TEST - Vibraforge")
        print(f"{'='*60}")
        print("1 - Test manuel (commandes individuelles)")
        print("2 - Test d'une adresse spécifique avec tous les seuils")
        print("3 - Test de toutes les adresses (balayage complet)")
        print("0 - Quitter")
        print(f"{'='*60}")
        
        choice = input("Choisir le mode (0-3): ").strip()
        
        if choice == '1':
            await test_manual_mode(client)
        elif choice == '2':
            addr = int(input("Quelle adresse tester (0-127)? "))
            await test_single_address(client, addr)
        elif choice == '3':
            await test_all_addresses(client)
        elif choice == '0':
            print("Fermeture du programme...")
            break
        else:
            print("Option invalide")


async def main():
    devices = await BleakScanner.discover()
    for d in devices:
        print('device name = ', d.name)
        if d.name != None:
            if d.name == CONTROL_UNIT_NAME:
                print('central unit BLE found!!!')
                async with BleakClient(d.address) as client:
                    print(f'BLE connected to {d.address}')
                    val = await client.read_gatt_char(CHARACTERISTIC_UUID)
                    print('Motor read = ', val)
                    await setMotor(client)


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Read CHARACTERISTIC_UUID and CONTROL_UNIT_NAME from the command line.")

    # Add arguments with flags
    parser.add_argument(
        "-uuid", "--characteristic_uuid", required=False, type=str,
        default="f22535de-5375-44bd-8ca9-d0ea9ff9e410",
        help="The UUID of the characteristic"
    )
    parser.add_argument(
        "-name", "--control_unit_name", required=False, type=str, 
        default="QT Py ESP32-S3",
        help="The Bluetooth name of the control unit"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access and print the parameters
    print(f"CHARACTERISTIC_UUID: {args.characteristic_uuid}")
    print(f"CONTROL_UNIT_NAME: {args.control_unit_name}")

    CHARACTERISTIC_UUID = args.characteristic_uuid
    CONTROL_UNIT_NAME = args.control_unit_name
    
    asyncio.run(main())
