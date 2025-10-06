#!/usr/bin/env python3
"""
Example of how to read and use Protocol Buffer data.
"""

# Import the generated protobuf classes
import importlib.util
import sys
import os
import struct
from pathlib import Path
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# Ensure demo encryption key is available via environment variable if none provided externally.
# OEM key supplied by user (16-byte AES-128 key: 2fb8344d08a5f00923c365e542be49d6)
os.environ.setdefault("PROTO_ENCRYPTION_KEY", "2fb8344d08a5f00923c365e542be49d6")


def round_to_multiple_of_16(value: int) -> int:
    """
    Round a value to the nearest multiple of 16.
    
    Arguments:
        value (int): The integer value to round.
    
    Returns:
        int: The rounded value which is a multiple of 16.
    """
    # Round up to the next multiple of 16
    return ((value + 15) // 16) * 16


def _load_encryption_key() -> bytes:
    """
    Load AES key from env variables or fall back to OEM key.
    
    Arguments:
        None
    
    Returns:
        bytes: The AES key (16, 24, or 32 bytes for AES-128/192/256).
    """
    # Try to load key directly from environment variable
    raw_key = os.environ.get("PROTO_ENCRYPTION_KEY")
    
    # If key provided directly, use it if valid length
    if raw_key:
        # Get bytes 
        candidate = raw_key.encode("utf-8")
        # Check length
        if len(candidate) in {16, 24, 32}:
            # Valid key length
            return candidate

    
    # Try to load key from file if specified
    key_file_path = os.environ.get("PROTO_ENCRYPTION_KEY_FILE")
    
    # If there is a file path, try to read it
    if key_file_path:
        try:
            # Read file bytes
            file_bytes = Path(key_file_path).read_bytes()
            # Check length
            if len(file_bytes) in {16, 24, 32}:
                # Return valid key
                return file_bytes
            
        # If not found or error reading, ignore
        except OSError:
            pass




# LOAD THE ENCRYPTION KEY
ENCRYPTION_KEY_BYTES = _load_encryption_key()



# LOAD THE PROTOBUF MODULE

# Dynamically load the raw_data protobuf module
spec = importlib.util.spec_from_file_location("raw_data_pb2", os.path.join(os.path.dirname(__file__), "raw_data_pb2.py"))
# Load the raw_data protobuf module
raw_data_pb2 = importlib.util.module_from_spec(spec)
# Register the module in sys.modules
sys.modules["raw_data_pb2"] = raw_data_pb2
# Execute the module
spec.loader.exec_module(raw_data_pb2)


def create_waveform_data() -> dict:
    """
    Create example waveform data using raw_data protobuf messages.
    
    Arguments:
        None
    
    Returns:
        dict: Dictionary with 'waveform' key containing the Waveform protobuf message.
    """
    
    # Create PPG object
    ppg = raw_data_pb2.Ppg()
    ppg.samplingRate = 100  # 100 Hz
    
    # Add some sample PPG channels
    for i in range(5):
        sample = ppg.ppg_samples.add()
        sample.red = 1000 + i * 10
        sample.ir = 800 + i * 8
        sample.green = 600 + i * 6
    
    # Create IMU data
    imu = raw_data_pb2.Imu()
    imu.samplingRate = 50  # 50 Hz
    
    # Add some acceleration samples
    for i in range(3):
        acc = imu.acc.add()
        acc.norm = 1.0 + i * 0.1
    
    # Create waveform object including everything
    waveform = raw_data_pb2.Waveform()
    waveform.epoch = 1696204800  
    waveform.ppg.CopyFrom(ppg)
    waveform.imu.CopyFrom(imu)
    waveform.heartRate.value = 72
    waveform.oxigenSaturation.value = 98
    waveform.arterialBloodPressure.systolic = 120
    waveform.arterialBloodPressure.diastolic = 80
    waveform.skinTemperature.value = 36.5
    
    return {
        'waveform': waveform
    }


def serialize_data(data: dict) -> tuple[int, dict]:
    """
    Serialize protobuf data to bytes.
    
    Arguments:
        data (dict): Dictionary of protobuf messages to serialize.
    
    Returns:
        tuple: (original payload length, serialized data dictionary)
    """
    # Local dictionary to hold serialized data
    serialized = {}
    proto_data_length = 0

    # Serialize each protobuf message to bytes
    for key, message in data.items():
        # Serialize the message
        serialized[key] = message.SerializeToString()
        # Get the original length
        proto_data_length = len(serialized[key])
        # Print size of original serialized message
        print(f"{key}: {proto_data_length} bytes")
        
    # Return the serialized data dictionary
    return proto_data_length, serialized


def _pack_serialized_data(serialized_data: dict) -> bytes:
    """
    Pack serialized protobuf messages into a single byte payload.
    
    Arguments:
        serialized_data (dict): Dictionary of serialized protobuf messages.
    
    Returns:
        bytes: The packed byte payload.
    """
    # Local bytearray to hold the packed payload
    payload = bytearray()
    
    # Pack each entry as: [key length (1 byte)][key bytes][data length (2 bytes)][data bytes]
    for key, blob in serialized_data.items():
        # Encode key to bytes
        key_bytes = key.encode("utf-8")
        
        # Validate lengths
        if len(key_bytes) > 255:
            # Key too long for 1-byte length
            raise ValueError("Key length exceeds 255 bytes; cannot pack into unsigned int 8")
        
        # Append key length
        payload.extend(struct.pack("<B", len(key_bytes)))
        
        # Append key bytes
        payload.extend(key_bytes)
        
        # Check blob length
        if len(blob) > 0xFFFF:
            raise ValueError("Serialized proto message exceeds 65535 bytes; cannot pack length into 2 bytes")

        # Append data length (2 bytes)
        payload.extend(struct.pack("<H", len(blob)))

        # Append data bytes
        payload.extend(blob)
        
    # Return the packed payload as bytes
    return bytes(payload)


def _unpack_serialized_data(payload: bytes, proto_data_length: int) -> dict:
    """
    Unpack a payload into the serialized protobuf message dictionary.
    
    Arguments:
        payload (bytes): The packed byte payload.
        proto_data_length (int): The expected length of the original unencrypted payload.
        
    Returns:
        dict: Dictionary of serialized protobuf messages.
    """
    # Local dictionary to hold unpacked serialized data
    offset = 0
    serialized_data = {}
    
    # Length of the payload
    total_length = len(payload)
    
    # Unpack until we reach the end of the payload
    while offset < total_length - proto_data_length + 1:
        # Read key length
        if offset + 1 > total_length:
            # Not enough bytes for key length
            raise ValueError("Corrupted payload: missing key length header")
        # Read key length
        key_length = struct.unpack_from("<B", payload, offset)[0]
        # Move offset forward
        offset += 1

        # Read key bytes
        if offset + key_length > total_length:
            # Not enough bytes for key bytes
            raise ValueError("Corrupted payload: incomplete key bytes")
        # Read key bytes
        key = payload[offset:offset + key_length].decode("utf-8")
        # Move offset forward
        offset += key_length

        # Read data length
        if offset + 2 > total_length:
            # Not enough bytes for data length
            raise ValueError("Corrupted payload: missing data length header")
        # Read data length
        #data_length = proto_data_length
        data_length = struct.unpack_from("<H", payload, offset)[0]
        # Move offset forward
        offset += 2

        # Read data bytes
        if offset + data_length > total_length:
            # Not enough bytes for data bytes
            raise ValueError("Corrupted payload: incomplete data bytes")
        # Read data bytes
        serialized_data[key] = payload[offset:offset + data_length]
        # Move offset forward
        offset += data_length

    # Return the unpacked serialized data dictionary
    return serialized_data


def encrypt_serialized_data(serialized_data: dict) -> tuple[bytes, bytes]:
    """
    Encrypt serialized protobuf messages using AES-CBC with PKCS#7 padding.
    
    Arguments:
        serialized_data (dict): Dictionary of serialized protobuf messages.
    
    Returns:
        tuple: (IV bytes, ciphertext bytes)
    """
    # Pack the serialized data into a single payload
    payload = _pack_serialized_data(serialized_data)
    
    # Encrypt the payload using AES-CBC with PKCS#7 padding
    # Generate a random 16-byte IV
    iv = get_random_bytes(16)
    # Create AES cipher
    cipher = AES.new(ENCRYPTION_KEY_BYTES, AES.MODE_CBC, iv=iv)
    # Pad the payload
    payload_padded = pad(payload, AES.block_size)
    # Encrypt the payload
    ciphertext = cipher.encrypt(payload_padded)

    # Return original payload length, IV, and ciphertext
    return iv, ciphertext


def decrypt_serialized_data(iv: bytes, proto_data_length: int, ciphertext: bytes) -> dict:
    """
    Decrypt AES-CBC payload back into serialized protobuf messages.
    
    Arguments:
        iv (bytes): The 16-byte AES IV used for decryption.
        ciphertext (bytes): The AES-CBC encrypted payload.

    Returns:
        dict: Dictionary of deserialized protobuf messages.
    """
    # Decrypt the ciphertext using AES-CBC with PKCS#7 unpadding
    cipher = AES.new(ENCRYPTION_KEY_BYTES, AES.MODE_CBC, iv=iv)
    
    # Decrypt the payload
    payload = cipher.decrypt(ciphertext)
    
    # Unpad the  payload
    payload = unpad(payload, AES.block_size)

    # Unpack the payload back into serialized data dictionary
    return _unpack_serialized_data(payload, proto_data_length)



def deserialize_data(serialized_data: dict) -> dict:
    """
    Deserialize bytes back to protobuf messages.
    
    Arguments:
        serialized_data (dict): Dictionary of serialized protobuf messages.
    
    Returns:
        dict: Dictionary of deserialized protobuf messages.
    """
    # Local dictionary to hold deserialized message instances
    instances = {}
    
    # Create new message instances for waveform data
    if 'waveform' in serialized_data:
        # Create new Waveform instance
        waveform = raw_data_pb2.Waveform()
        # Parse the serialized data into the waveform instance
        waveform.ParseFromString(serialized_data['waveform'])
        # Add the waveform instance to the local dictionary
        instances['waveform'] = waveform
    
    # Return the instances dictionary
    return instances


def print_waveform_data(data: dict) -> None:
    """
    Print waveform data in a readable format.
    
    Arguments:
        data (dict): Dictionary containing the 'waveform' protobuf message.
    
    Returns:
        None
    """
    
    # If waveform data is present, print its contents
    if 'waveform' in data:
        wf = data['waveform']
        print(f"Epoch: {wf.epoch}")
        print(f"PPG Sampling Rate: {wf.ppg.samplingRate} Hz")
        print(f"PPG Samples: {len(wf.ppg.ppg_samples)} samples")
        if len(wf.ppg.ppg_samples) > 0:
            sample = wf.ppg.ppg_samples[0]
            print(f"  First sample - Red: {sample.red}, IR: {sample.ir}, Green: {sample.green}")
        print(f"IMU Sampling Rate: {wf.imu.samplingRate} Hz")
        print(f"IMU Acceleration samples: {len(wf.imu.acc)}")
        if len(wf.imu.acc) > 0:
            print(f"  First acceleration norm: {wf.imu.acc[0].norm}")
        print(f"Heart Rate: {wf.heartRate.value} bpm")
        print(f"Oxygen Saturation: {wf.oxigenSaturation.value} %")
        print(f"Blood Pressure: {wf.arterialBloodPressure.systolic}/{wf.arterialBloodPressure.diastolic} mmHg")
        print(f"Skin Temperature: {wf.skinTemperature.value} °C")
    

def save_to_file(serialized_data: dict, proto_data_length: int, filename: str) -> None:
    """
    Encrypt and save serialized data to a binary file.
    
    Arguments:
        serialized_data (dict): Dictionary of serialized protobuf messages.
        proto_data_length (int): The expected length of the original unencrypted payload.
        filename (str): The path to the output binary file.
        
    Returns:
        None
    """
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Encrypt the serialized data
    iv, ciphertext = encrypt_serialized_data(serialized_data)
    
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


def load_from_file(filename: str) -> tuple[int, bytes, bytes]:
    """
    Load and decrypt serialized data from a binary file.
    
    Arguments:
        filename (str): The path to the input binary file.
        
    Returns:
        tuple: (original payload length, IV bytes, ciphertext bytes)
    """
    
    # Read from file
    with open(filename, 'rb') as f:
        # Read IV
        iv = f.read(16)
        
        # Check IV length
        if len(iv) != 16:
            raise ValueError("Encrypted file missing 16-byte IV header")
        
        # Read Proto data length
        proto_data_length_bytes = f.read(2)

        # Check Proto data length header
        if len(proto_data_length_bytes) != 2:
            raise ValueError("Encrypted file missing proto data length header")

        # Unpack Proto data length
        original_length = struct.unpack('<H', proto_data_length_bytes)[0]

        # Read remaining ciphertext (everything else in the file)
        ciphertext = f.read()

        # Check we have some ciphertext
        if len(ciphertext) == 0:
            raise ValueError("Encrypted file has no ciphertext data")
        
    # Decrypt and unpack
    return decrypt_serialized_data(iv, original_length, ciphertext)


def main() -> None:
    """
    Main function demonstrating proto file usage.
    
    Arguments:
        None
        
    Returns:
        None
    """
    # Print header
    print("\n=== Protocol Buffer Example ===")
    
    # 1. Create waveform data
    print("\n1. Creating waveform data...")
    waveform_data = create_waveform_data()
    print_waveform_data(waveform_data)
    
    # 2. Serialize to bytes
    print("\n2. Serializing waveform data...")
    proto_data_length, serialized = serialize_data(waveform_data)
    
    # 3. Save to file
    print("\n3. Encrypting and saving to file...")
    save_to_file(serialized, proto_data_length, "src/bin/ONAVITAL_raw_data.bin")
    print("Data saved to src/bin/ONAVITAL_raw_data.bin")

    # 4. Load from file
    print("\n4. Loading and decrypting from file...")
    loaded_serialized = load_from_file("src/bin/ONAVITAL_raw_data.bin")
    print("Data loaded from src/bin/ONAVITAL_raw_data.bin")

    # 5. Deserialize back to objects
    print("\n5. Deserializing data...")
    loaded_data = deserialize_data(loaded_serialized)
    print_waveform_data(loaded_data)
    
    print("\n=== Example Complete ===\n")


if __name__ == "__main__":
    main()