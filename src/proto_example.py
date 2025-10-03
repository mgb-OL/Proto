#!/usr/bin/env python3
"""
Example of how to read and use Protocol Buffer data.
"""

# Import the generated protobuf classes
# Note: The generated file has a space in the name, so we import it as a module
import importlib.util
import sys
import os
import struct
from pathlib import Path
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# Ensure demo encryption key is available via environment variable if none provided externally.
os.environ.setdefault("PROTO_ENCRYPTION_KEY", "2fb8344d08a5f00923c365e542be49d6")


def round_to_multiple_of_16(value):
    """Round a value to the nearest multiple of 16."""
    return ((value + 15) // 16) * 16


def _load_encryption_key() -> bytes:
    """Load AES key from env variables or fall back to OEM key."""
    raw_key = os.environ.get("PROTO_ENCRYPTION_KEY")
    if raw_key:
        candidate = raw_key.encode("utf-8")
        if len(candidate) in {16, 24, 32}:
            return candidate

    key_file_path = os.environ.get("PROTO_ENCRYPTION_KEY_FILE")
    if key_file_path:
        try:
            file_bytes = Path(key_file_path).read_bytes()
            if len(file_bytes) in {16, 24, 32}:
                return file_bytes
        except OSError:
            pass

    # OEM key supplied by user (16-byte AES-128 key: 2fb8344d08a5f00923c365e542be49d6)
    return bytes.fromhex("2fb8344d08a5f00923c365e542be49d6")


ENCRYPTION_KEY_BYTES = _load_encryption_key()

# Load the generated protobuf module
spec = importlib.util.spec_from_file_location("alg 1_pb", os.path.join(os.path.dirname(__file__), "alg 1_pb.py"))
health_pb2 = importlib.util.module_from_spec(spec)
sys.modules["alg 1_pb"] = health_pb2
spec.loader.exec_module(health_pb2)


def create_health_data():
    """Create example health data using protobuf messages."""
    
    # Create heart rate data
    heart_rate = health_pb2.HeartRate()
    heart_rate.value = 72
    
    # Create oxygen saturation data
    oxygen_saturation = health_pb2.OxigenSaturation()
    oxygen_saturation.value = 98
    
    # Create blood pressure data
    blood_pressure = health_pb2.ArterialBloodPressure()
    blood_pressure.systolic = 120
    blood_pressure.diastolic = 80
    
    # Create skin temperature data
    skin_temp = health_pb2.SkinTemperature()
    skin_temp.value = 36.5
    
    # Create battery level data
    battery = health_pb2.BatteryLevel()
    battery.value = 85
    
    # Create steps counter data
    steps = health_pb2.StepsCounter()
    steps.epochInit = 1696204800  # Unix timestamp
    steps.steps = 8547
    
    return {
        'heart_rate': heart_rate,
        'oxygen_saturation': oxygen_saturation,
        'blood_pressure': blood_pressure,
        'skin_temperature': skin_temp,
        'battery': battery,
        'steps': steps
    }


def serialize_data(data):
    """Serialize protobuf data to bytes."""
    serialized = {}
    for key, message in data.items():
        serialized[key] = message.SerializeToString()
        print(f"{key}: {len(serialized[key])} bytes")
    return serialized


def _pack_serialized_data(serialized_data):
    """Pack serialized protobuf messages into a single byte payload."""
    payload = bytearray()
    for key, blob in serialized_data.items():
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 255:
            raise ValueError("Key length exceeds 255 bytes; cannot pack into unsigned int 8")
        payload.extend(struct.pack("<B", len(key_bytes)))
        payload.extend(key_bytes)
        if len(blob) > 0xFFFF:
            raise ValueError("Serialized proto message exceeds 65535 bytes; cannot pack length into 2 bytes")
        payload.extend(struct.pack("<H", len(blob)))
        payload.extend(blob)
    return bytes(payload)


def _unpack_serialized_data(payload):
    """Unpack a payload into the serialized protobuf message dictionary."""
    offset = 0
    total_length = len(payload)
    serialized_data = {}
    while offset < total_length:
        if offset + 1 > total_length:
            raise ValueError("Corrupted payload: missing key length header")
        key_length = struct.unpack_from("<B", payload, offset)[0]
        offset += 1

        if offset + key_length > total_length:
            raise ValueError("Corrupted payload: incomplete key bytes")
        key = payload[offset:offset + key_length].decode("utf-8")
        offset += key_length

        if offset + 2 > total_length:
            raise ValueError("Corrupted payload: missing data length header")
        data_length = struct.unpack_from("<H", payload, offset)[0]
        offset += 2

        if offset + data_length > total_length:
            raise ValueError("Corrupted payload: incomplete data bytes")
        serialized_data[key] = payload[offset:offset + data_length]
        offset += data_length

    return serialized_data


def encrypt_serialized_data(serialized_data):
    """Encrypt serialized protobuf messages using AES-CBC with PKCS#7 padding."""
    payload = _pack_serialized_data(serialized_data)
    iv = get_random_bytes(16)
    cipher = AES.new(ENCRYPTION_KEY_BYTES, AES.MODE_CBC, iv=iv)
    ciphertext = cipher.encrypt(pad(payload, AES.block_size))
    return len(payload),iv, ciphertext


def decrypt_serialized_data(iv, ciphertext):
    """Decrypt AES-CBC payload back into serialized protobuf messages."""
    cipher = AES.new(ENCRYPTION_KEY_BYTES, AES.MODE_CBC, iv=iv)
    payload = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return _unpack_serialized_data(payload)


def deserialize_data(serialized_data):
    """Deserialize bytes back to protobuf messages."""
    # Create new message instances
    heart_rate = health_pb2.HeartRate()
    oxygen_saturation = health_pb2.OxigenSaturation()
    blood_pressure = health_pb2.ArterialBloodPressure()
    skin_temp = health_pb2.SkinTemperature()
    battery = health_pb2.BatteryLevel()
    steps = health_pb2.StepsCounter()
    
    # Parse from bytes
    heart_rate.ParseFromString(serialized_data['heart_rate'])
    oxygen_saturation.ParseFromString(serialized_data['oxygen_saturation'])
    blood_pressure.ParseFromString(serialized_data['blood_pressure'])
    skin_temp.ParseFromString(serialized_data['skin_temperature'])
    battery.ParseFromString(serialized_data['battery'])
    steps.ParseFromString(serialized_data['steps'])
    
    return {
        'heart_rate': heart_rate,
        'oxygen_saturation': oxygen_saturation,
        'blood_pressure': blood_pressure,
        'skin_temperature': skin_temp,
        'battery': battery,
        'steps': steps
    }


def print_health_data(data):
    """Print health data in a readable format."""
    print("\n=== Health Data ===")
    print(f"Heart Rate: {data['heart_rate'].value} bpm")
    print(f"Oxygen Saturation: {data['oxygen_saturation'].value}%")
    print(f"Blood Pressure: {data['blood_pressure'].systolic}/{data['blood_pressure'].diastolic} mmHg")
    print(f"Skin Temperature: {data['skin_temperature'].value}°C")
    print(f"Battery Level: {data['battery'].value}%")
    print(f"Steps: {data['steps'].steps} (since epoch {data['steps'].epochInit})")


def save_to_file(serialized_data, filename):
    """
    Encrypt and save serialized data to a binary file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Encrypt the serialized data
    proto_data_length, iv, ciphertext = encrypt_serialized_data(serialized_data)
    
    # Write to file
    with open(filename, 'wb') as f:
        # Check IV length
        if len(iv) != 16:
            raise ValueError("Expected 16-byte IV for AES-CBC encryption")
        
        # Write IV directly (16 bytes)
        f.write(iv)
        
        # Write proto data length
        f.write(struct.pack('<H', proto_data_length))
        
        # Write the ciphertext
        f.write(ciphertext)


def load_from_file(filename):
    """
    Load and decrypt serialized data from a binary file.
    """
    # Read from file
    with open(filename, 'rb') as f:
        # Read IV
        iv = f.read(16)
        
        # Check IV length
        if len(iv) != 16:
            raise ValueError("Encrypted file missing 16-byte IV header")
        
        # Read Proto data length
        proto_data_length = f.read(2)

        # Check Proto data length header
        if len(proto_data_length) != 2:
            raise ValueError("Encrypted file missing proto data length header")

        # Unpack Proto data length
        encrypted_length = round_to_multiple_of_16(struct.unpack('<H', proto_data_length)[0])

        # Read ciphertext
        ciphertext = f.read(encrypted_length)

        # Check ciphertext length
        if len(ciphertext) != encrypted_length:
            raise ValueError("Encrypted file has incomplete ciphertext data")
        
    # Decrypt and unpack
    return decrypt_serialized_data(iv, ciphertext)


def main():
    """Main function demonstrating proto file usage."""
    print("=== Protocol Buffer Example ===")
    
    # 1. Create health data
    print("\n1. Creating health data...")
    health_data = create_health_data()
    print_health_data(health_data)
    
    # 2. Serialize to bytes
    print("\n2. Serializing data...")
    serialized = serialize_data(health_data)
    
    # 3. Save to file
    print("\n3. Encrypting and saving to file...")
    save_to_file(serialized, "src/bin/health_data.bin")
    print("Data saved to src/bin/health_data.bin")

    # 4. Load from file
    print("\n4. Loading and decrypting from file...")
    loaded_serialized = load_from_file("src/bin/health_data.bin")
    print("Data loaded from src/bin/health_data.bin")

    # 5. Deserialize back to objects
    print("\n5. Deserializing data...")
    loaded_data = deserialize_data(loaded_serialized)
    print_health_data(loaded_data)
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()