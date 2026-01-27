#!/usr/bin/env python3
"""
VibraForge BLE tester — version "fusion" (menu interactif + robustesse)

Objectifs
- Menu interactif (pratique pour explorer / tester)
- Adressage robuste: global_addr -> (chain, local) via units_per_chain (par défaut 16)
- Validations (addr/mode/duty/freq)
- Sélection automatique du device au meilleur RSSI si plusieurs matchent
- Sécurité: STOP de l’adresse utilisée en sortie + option STOP ALL

Dépendances
  pip install bleak

Exemples (non-interactif)
  python vibraforge_test_fusion.py --name "QT Py ESP32-S3" --pulse --addr 0 --duty 10 --freq 3 --duration 1.0
  python vibraforge_test_fusion.py --stop-all

Mode interactif (menu)
  python vibraforge_test_fusion.py
"""

import argparse
import asyncio
from dataclasses import dataclass
from typing import Optional, List, Tuple

from bleak import BleakScanner, BleakClient


DEFAULT_CONTROL_UNIT_NAME = "QT Py ESP32-S3"
DEFAULT_CHARACTERISTIC_UUID = "f22535de-5375-44bd-8ca9-d0ea9ff9e410"

# Deux encodages possibles pour l’octet 3, car les docs/code existants divergent parfois.
# - "freq3" (défaut): 1 DDDD FFF  -> duty (0..15), freq (0..7)
# - "freq2wave":       1 DDDD FF W -> duty (0..15), freq (0..3), wave (0..1)
ENCODINGS = ("freq3", "freq2wave")


@dataclass
class VFConfig:
    name: str = DEFAULT_CONTROL_UNIT_NAME
    char_uuid: str = DEFAULT_CHARACTERISTIC_UUID
    chains: int = 4
    units_per_chain: int = 16
    scan_timeout: float = 6.0
    with_response: bool = False
    encoding: str = "freq3"
    default_duty: int = 8
    default_freq: int = 3
    default_wave: int = 0
    default_duration: float = 1.0
    default_cooldown: float = 0.2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VibraForge BLE tester (fusion)")
    p.add_argument("--name", default=DEFAULT_CONTROL_UNIT_NAME, help="Nom Bluetooth du contrôle (advertised name)")
    p.add_argument("--uuid", default=DEFAULT_CHARACTERISTIC_UUID, help="UUID de la caractéristique GATT")
    p.add_argument("--chains", type=int, default=4, help="Nombre de chaînes (serial groups)")
    p.add_argument("--units-per-chain", type=int, default=16, help="Nombre d’unités par chaîne")
    p.add_argument("--scan-timeout", type=float, default=6.0, help="Durée du scan BLE (s)")
    p.add_argument("--with-response", action="store_true", help="Write GATT avec response=True (si supporté)")
    p.add_argument("--encoding", choices=ENCODINGS, default="freq3",
                   help="Encodage de l’octet 3 (freq/duty/wave)")

    # Mode non-interactif (optionnel)
    p.add_argument("--addr", type=int, default=None, help="Adresse globale (0..chains*units_per_chain-1)")
    p.add_argument("--start", action="store_true", help="Démarrer (mode=1) sur --addr")
    p.add_argument("--stop", action="store_true", help="Arrêter (mode=0) sur --addr")
    p.add_argument("--pulse", action="store_true", help="Démarrer puis arrêter après --duration sur --addr")
    p.add_argument("--stop-all", action="store_true", help="Envoie STOP sur toutes les adresses")
    p.add_argument("--duty", type=int, default=8, help="Puissance duty (0..15)")
    p.add_argument("--freq", type=int, default=3, help="Fréquence (freq3: 0..7, freq2wave: 0..3)")
    p.add_argument("--wave", type=int, default=0, help="Wave (uniquement encoding=freq2wave) 0..1")
    p.add_argument("--duration", type=float, default=1.0, help="Durée (s) pour --pulse")
    p.add_argument("--cooldown", type=float, default=0.2, help="Pause (s) après STOP (utile)")

    return p


def _max_addr(cfg: VFConfig) -> int:
    return cfg.chains * cfg.units_per_chain - 1


def validate_addr(cfg: VFConfig, addr: int) -> None:
    if not isinstance(addr, int):
        raise ValueError("addr doit être un entier")
    if addr < 0 or addr > _max_addr(cfg):
        raise ValueError(f"addr hors limites: {addr} (attendu 0..{_max_addr(cfg)})")


def validate_duty(duty: int) -> None:
    if duty < 0 or duty > 15:
        raise ValueError(f"duty hors limites: {duty} (attendu 0..15)")


def validate_freq(cfg: VFConfig, freq: int) -> None:
    if cfg.encoding == "freq3":
        if freq < 0 or freq > 7:
            raise ValueError(f"freq hors limites: {freq} (attendu 0..7 pour encoding=freq3)")
    else:
        if freq < 0 or freq > 3:
            raise ValueError(f"freq hors limites: {freq} (attendu 0..3 pour encoding=freq2wave)")


def validate_wave(cfg: VFConfig, wave: int) -> None:
    if cfg.encoding != "freq2wave":
        return
    if wave < 0 or wave > 1:
        raise ValueError(f"wave hors limites: {wave} (attendu 0..1)")


def split_addr(cfg: VFConfig, global_addr: int) -> Tuple[int, int]:
    """global -> (chain, local)"""
    validate_addr(cfg, global_addr)
    chain = global_addr // cfg.units_per_chain
    local = global_addr % cfg.units_per_chain
    return chain, local


def build_command_bytes(cfg: VFConfig, global_addr: int, mode: int, duty: int, freq: int, wave: int = 0) -> bytes:
    """
    Forme un paquet de 60 octets:
      byte0: 00CCCC0M   (C = chain/group, M = mode 0/1)
      byte1: 01LLLLLL   (L = local address)
      byte2: encoding:
            - freq3:    1DDDDFFF
            - freq2wave:1DDDDFFW
      bytes3..59: 0
    """
    if mode not in (0, 1):
        raise ValueError("mode doit être 0 (stop) ou 1 (start)")
    validate_duty(duty)
    validate_freq(cfg, freq)
    validate_wave(cfg, wave)

    chain, local = split_addr(cfg, global_addr)

    # byte0: 00CCCC0M (chain sur 4 bits)
    if chain < 0 or chain > 15:
        raise ValueError(f"chain hors 4 bits: {chain} (vérifie chains/units_per_chain)")
    b0 = (chain & 0x0F) << 1
    b0 |= (mode & 0x01)

    # byte1: 01LLLLLL (local sur 6 bits)
    if local < 0 or local > 63:
        raise ValueError(f"local hors 6 bits: {local}")
    b1 = 0x40 | (local & 0x3F)

    # byte2
    if cfg.encoding == "freq3":
        b2 = 0x80 | ((duty & 0x0F) << 3) | (freq & 0x07)
    else:
        b2 = 0x80 | ((duty & 0x0F) << 3) | ((freq & 0x03) << 1) | (wave & 0x01)

    payload = bytearray(60)
    payload[0] = b0
    payload[1] = b1
    payload[2] = b2
    return bytes(payload)


async def find_best_device_by_name(name: str, timeout: float) -> Optional[object]:
    """Retourne le device BLE dont le name matche exactement, avec le meilleur RSSI."""
    devices = await BleakScanner.discover(timeout=timeout)
    matches = [d for d in devices if (getattr(d, "name", None) or "") == name]

    if not matches:
        # fallback: contient le nom (utile si certains OS tronquent/varient)
        matches = [d for d in devices if name.lower() in ((getattr(d, "name", None) or "").lower())]

    if not matches:
        return None

    # Trier par RSSI (plus grand = plus proche, souvent)
    matches.sort(key=lambda d: (getattr(d, "rssi", -999)), reverse=True)
    return matches[0]


async def write_command(client: BleakClient, cfg: VFConfig, data: bytes) -> None:
    await client.write_gatt_char(cfg.char_uuid, data, response=cfg.with_response)


async def stop_all(client: BleakClient, cfg: VFConfig, duty_for_stop: int = 0, freq_for_stop: int = 0) -> None:
    for addr in range(0, _max_addr(cfg) + 1):
        data = build_command_bytes(cfg, addr, mode=0, duty=duty_for_stop, freq=freq_for_stop, wave=0)
        await write_command(client, cfg, data)
        await asyncio.sleep(0.01)


async def pulse(client: BleakClient, cfg: VFConfig, addr: int, duty: int, freq: int, wave: int, duration: float, cooldown: float) -> None:
    start_data = build_command_bytes(cfg, addr, mode=1, duty=duty, freq=freq, wave=wave)
    stop_data = build_command_bytes(cfg, addr, mode=0, duty=duty, freq=freq, wave=wave)
    await write_command(client, cfg, start_data)
    await asyncio.sleep(max(0.0, float(duration)))
    await write_command(client, cfg, stop_data)
    await asyncio.sleep(max(0.0, float(cooldown)))


def _prompt_int(msg: str, default: Optional[int] = None) -> int:
    while True:
        raw = input(msg).strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Entrée invalide. Merci d’entrer un nombre entier.")


def _prompt_float(msg: str, default: Optional[float] = None) -> float:
    while True:
        raw = input(msg).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Entrée invalide. Merci d’entrer un nombre (ex: 1.5).")


async def interactive_menu(client: BleakClient, cfg: VFConfig) -> None:
    last_addr: Optional[int] = None

    while True:
        print("\n--- MENU VibraForge ---")
        print("1) Démarrer/Arrêter une adresse (commande simple)")
        print("2) Pulse: démarrer puis arrêter après une durée")
        print("3) Sweep: tester une plage d’adresses (pulse sur chaque)")
        print("4) STOP ALL (coupe toutes les adresses)")
        print("5) Modifier les valeurs par défaut (duty/freq/wave/duration)")
        print("0) Quitter")
        print("6) Test progressif d'intensité (0..15) sur une adresse")
        choice = input("Choix: ").strip()

        if choice == "0":
            print("Sortie du menu.")
            return

        if choice == "5":
            cfg.default_duty = _prompt_int(f"Duty par défaut (0..15) [{cfg.default_duty}]: ", cfg.default_duty)
            validate_duty(cfg.default_duty)
            cfg.default_freq = _prompt_int(f"Freq par défaut [{cfg.default_freq}]: ", cfg.default_freq)
            validate_freq(cfg, cfg.default_freq)
            if cfg.encoding == "freq2wave":
                cfg.default_wave = _prompt_int(f"Wave par défaut (0..1) [{cfg.default_wave}]: ", cfg.default_wave)
                validate_wave(cfg, cfg.default_wave)
            cfg.default_duration = _prompt_float(f"Durée pulse (s) [{cfg.default_duration}]: ", cfg.default_duration)
            cfg.default_cooldown = _prompt_float(f"Cooldown après STOP (s) [{cfg.default_cooldown}]: ", cfg.default_cooldown)
            print("OK, valeurs mises à jour.")
            continue

        if choice == "4":
            print("Envoi STOP ALL...")
            await stop_all(client, cfg)
            print("STOP ALL terminé.")
            last_addr = None
            continue

        if choice in ("1", "2"):
            addr = _prompt_int(f"Adresse globale (0..{_max_addr(cfg)}): ")
            validate_addr(cfg, addr)
            duty = _prompt_int(f"Duty (0..15) [{cfg.default_duty}]: ", cfg.default_duty)
            validate_duty(duty)
            freq = _prompt_int(f"Freq [{cfg.default_freq}]: ", cfg.default_freq)
            validate_freq(cfg, freq)
            wave = cfg.default_wave
            if cfg.encoding == "freq2wave":
                wave = _prompt_int(f"Wave (0..1) [{cfg.default_wave}]: ", cfg.default_wave)
                validate_wave(cfg, wave)

            if choice == "1":
                mode = _prompt_int("Mode: 1=START, 0=STOP [1]: ", 1)
                if mode not in (0, 1):
                    print("Mode invalide (attendu 0 ou 1).")
                    continue
                data = build_command_bytes(cfg, addr, mode=mode, duty=duty, freq=freq, wave=wave)
                await write_command(client, cfg, data)
                chain, local = split_addr(cfg, addr)
                print(f"OK envoyé (addr={addr} -> chain={chain}, local={local}, mode={mode}, duty={duty}, freq={freq}, wave={wave})")
                last_addr = addr
                continue

            if choice == "2":
                duration = _prompt_float(f"Durée (s) [{cfg.default_duration}]: ", cfg.default_duration)
                cooldown = _prompt_float(f"Cooldown (s) [{cfg.default_cooldown}]: ", cfg.default_cooldown)
                await pulse(client, cfg, addr, duty, freq, wave, duration, cooldown)
                chain, local = split_addr(cfg, addr)
                print(f"Pulse OK (addr={addr} -> chain={chain}, local={local}, duty={duty}, freq={freq}, wave={wave}, duration={duration})")
                last_addr = addr
                continue

        if choice == "3":
            start_addr = _prompt_int(f"Début plage (0..{_max_addr(cfg)}): ")
            end_addr = _prompt_int(f"Fin plage (0..{_max_addr(cfg)}): ")
            if start_addr > end_addr:
                start_addr, end_addr = end_addr, start_addr
            validate_addr(cfg, start_addr)
            validate_addr(cfg, end_addr)

            duty = _prompt_int(f"Duty (0..15) [{cfg.default_duty}]: ", cfg.default_duty)
            validate_duty(duty)
            freq = _prompt_int(f"Freq [{cfg.default_freq}]: ", cfg.default_freq)
            validate_freq(cfg, freq)
            wave = cfg.default_wave
            if cfg.encoding == "freq2wave":
                wave = _prompt_int(f"Wave (0..1) [{cfg.default_wave}]: ", cfg.default_wave)
                validate_wave(cfg, wave)

            duration = _prompt_float(f"Durée pulse (s) [{cfg.default_duration}]: ", cfg.default_duration)
            cooldown = _prompt_float(f"Cooldown (s) [{cfg.default_cooldown}]: ", cfg.default_cooldown)

            print(f"Sweep {start_addr}..{end_addr} (pulse sur chaque)")
            for addr in range(start_addr, end_addr + 1):
                await pulse(client, cfg, addr, duty, freq, wave, duration, cooldown)
                chain, local = split_addr(cfg, addr)
                print(f"  OK addr={addr} (chain={chain}, local={local})")
            last_addr = end_addr
            continue

        if choice == "6":
            addr = _prompt_int(f"Adresse cible (0..{_max_addr(cfg)}): ")
            validate_addr(cfg, addr)

            freq = _prompt_int(f"Freq [{cfg.default_freq}]: ", cfg.default_freq)
            validate_freq(cfg, freq)

            wave = cfg.default_wave
            if cfg.encoding == "freq2wave":
                wave = _prompt_int(f"Wave (0..1) [{cfg.default_wave}]: ", cfg.default_wave)
                validate_wave(cfg, wave)

            duration = _prompt_float(f"Durée par palier (s) [{cfg.default_duration}]: ", cfg.default_duration)

            print(f"\n--- Démarrage Ramp-up (0 -> 15) sur Address {addr} ---")
            for d in range(16):
                print(f" >> Test Intensité (Duty) {d}/15")
                # On utilise pulse pour faire : ON -> Attente(duration) -> OFF -> Attente(cooldown)
                # Cela permet de bien distinguer chaque palier
                await pulse(client, cfg, addr, duty=d, freq=freq, wave=wave, duration=duration, cooldown=0.3)

            print(f"--- Test progressif terminé sur {addr} ---")
            last_addr = addr
            continue

        print("Choix inconnu. Merci de choisir 0,1,2,3,4 ou 5.")


async def main_async(args: argparse.Namespace) -> None:
    cfg = VFConfig(
        name=args.name,
        char_uuid=args.uuid,
        chains=args.chains,
        units_per_chain=args.units_per_chain,
        scan_timeout=args.scan_timeout,
        with_response=args.with_response,
        encoding=args.encoding,
        default_duty=args.duty,
        default_freq=args.freq,
        default_wave=args.wave,
        default_duration=args.duration,
        default_cooldown=args.cooldown,
    )

    # Validations de base (pour éviter de se connecter pour rien)
    validate_duty(cfg.default_duty)
    validate_freq(cfg, cfg.default_freq)
    validate_wave(cfg, cfg.default_wave)

    print(f"Scan BLE (timeout={cfg.scan_timeout}s) pour trouver un device nommé: {cfg.name}")
    device = await find_best_device_by_name(cfg.name, timeout=cfg.scan_timeout)
    if device is None:
        raise RuntimeError(f"Aucun device trouvé avec le nom: {cfg.name}")

    dev_name = getattr(device, "name", "Unknown")
    dev_addr = getattr(device, "address", "Unknown")
    dev_rssi = getattr(device, "rssi", None)
    print(f"Device sélectionné: {dev_name} / {dev_addr}" + (f" / RSSI={dev_rssi}" if dev_rssi is not None else ""))

    last_addr_used: Optional[int] = None

    async with BleakClient(device) as client:
        print("Connecté.")

        try:
            if args.stop_all:
                print("STOP ALL demandé...")
                await stop_all(client, cfg)
                print("STOP ALL terminé.")
                return

            # Mode non-interactif si on a une action explicite
            non_interactive = any([args.start, args.stop, args.pulse]) or (args.addr is not None)
            if non_interactive and args.addr is None and not args.stop_all:
                raise ValueError("En mode non-interactif, il faut fournir --addr (sauf --stop-all).")

            if non_interactive and args.addr is not None:
                addr = args.addr
                validate_addr(cfg, addr)
                validate_duty(args.duty)
                validate_freq(cfg, args.freq)
                validate_wave(cfg, args.wave)

                # Si rien n'est demandé, on bascule menu
                if not any([args.start, args.stop, args.pulse]):
                    await interactive_menu(client, cfg)
                    return

                if args.pulse:
                    await pulse(client, cfg, addr, args.duty, args.freq, args.wave, args.duration, args.cooldown)
                    last_addr_used = addr
                    return

                if args.start:
                    data = build_command_bytes(cfg, addr, mode=1, duty=args.duty, freq=args.freq, wave=args.wave)
                    await write_command(client, cfg, data)
                    last_addr_used = addr
                    return

                if args.stop:
                    data = build_command_bytes(cfg, addr, mode=0, duty=args.duty, freq=args.freq, wave=args.wave)
                    await write_command(client, cfg, data)
                    last_addr_used = addr
                    return

            # Sinon menu
            await interactive_menu(client, cfg)

        finally:
            # Sécurité: STOP l’adresse utilisée si on en a une
            if last_addr_used is not None:
                try:
                    print(f"Sécurité: STOP addr={last_addr_used}")
                    data = build_command_bytes(cfg, last_addr_used, mode=0, duty=0, freq=0, wave=0)
                    await write_command(client, cfg, data)
                except Exception as e:
                    print(f"(Avertissement) STOP sécurité a échoué: {e}")


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
