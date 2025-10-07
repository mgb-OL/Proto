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


# Ensure demo encryption key is available via environment variable if none provided externally.
# OEM key supplied by user (16-byte AES-128)
os.environ.setdefault("PROTO_ENCRYPTION_KEY_FILE", "keys/OEM_KEY_COMPANY1_key_AES_CBC 1.bin")


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
    payload_decrypted = cipher.decrypt(ciphertext)
    
    # Extract payload using the proto_data_length
    payload = get_sub_byte_array(payload_decrypted, 0, proto_data_length)
    
    # Check if this looks like raw protobuf data (starts with typical protobuf bytes)
    if len(payload) > 0 and payload[0] in [0x08, 0x10, 0x12, 0x1a, 0x22]:  # Common protobuf field tags
        print("Detected raw protobuf data format")
        # Return raw protobuf data as 'waveform' key for compatibility
        return {'waveform': payload}
    
    # Otherwise try to unpack using our custom format
    try:
        return unpack_serialized_data(payload, proto_data_length)
    except (UnicodeDecodeError, ValueError) as e:
        print(f"Failed to unpack custom format: {e}")
        print("Treating as raw protobuf data")
        return {'waveform': payload}



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



def get_sub_byte_array(byte_array: bytes, start: int, length: int = None) -> bytes:
    """
    Extract a sub-byte array from a byte array.
    
    Arguments:
        byte_array (bytes): The source byte array.
        start (int): The starting index (0-based).
        length (int, optional): The number of bytes to extract. If None, extracts to the end.
    
    Returns:
        bytes: The extracted sub-byte array.
    
    Raises:
        ValueError: If start index is negative or out of bounds.
        ValueError: If length would exceed array bounds.
    """
    if start < 0:
        raise ValueError("Start index cannot be negative")
    
    if start >= len(byte_array):
        raise ValueError(f"Start index {start} is out of bounds for array of length {len(byte_array)}")
    
    if length is None:
        return byte_array[start:]
    
    if length < 0:
        raise ValueError("Length cannot be negative")
    
    end = start + length
    if end > len(byte_array):
        raise ValueError(f"End index {end} would exceed array bounds of {len(byte_array)}")
    
    return byte_array[start:end]



def load_from_file(filename: str) -> tuple[int, bytes, bytes]:
    """
    Load and decrypt serialized data from a binary file.
    
    Arguments:
        filename (str): The path to the input binary file.
        
    Returns:
        tuple: (original payload length, IV bytes, ciphertext bytes)
    """
    # Local bytearray to hold the IV
    iv = bytearray()
        
    # Read from file
    with open(filename, 'rb') as f:
        
        # Read IV
        iv_0 = f.read(4)
        iv_1 = f.read(4)
        iv_2 = f.read(4)
        iv_3 = f.read(4)

        # Combine IV parts
        iv.extend(iv_0)
        iv.extend(iv_1)
        iv.extend(iv_2)
        iv.extend(iv_3)

        # Check IV length
        if len(iv) != 16:
            raise ValueError("Encrypted file missing 16-byte IV header")
        
        # Read Proto data length
        proto_data_length_bytearray = f.read(2)
                
        # Check Proto data length header
        if len(proto_data_length_bytearray) != 2:
            raise ValueError("Encrypted file missing proto data length header")

        # Unpack Proto data length
        proto_data_length = struct.unpack('<H', proto_data_length_bytearray)[0]

        # Read remaining ciphertext (everything else in the file)
        ciphertext = f.read()
        
        # Check we have some ciphertext
        if len(ciphertext) == 0:
            raise ValueError("Encrypted file has no ciphertext data")
        
    # Decrypt and unpack
    return decrypt_serialized_data(iv, proto_data_length, ciphertext)



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



def unpack_serialized_data(payload: bytes, proto_data_length: int) -> dict:
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
    
    print(f"Unpacking payload: {total_length} bytes, expected proto length: {proto_data_length}")
    print(f"First 32 bytes: {payload[:32].hex()}")
    
    # Unpack until we reach the end of the payload
    while offset < total_length:
        # Read key length
        if offset + 1 > total_length:
            # Not enough bytes for key length
            raise ValueError("Corrupted payload: missing key length header")
        # Read key length
        key_length = struct.unpack_from("<B", payload, offset)[0]
        print(f"At offset {offset}: key_length = {key_length}")
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
        data_length = struct.unpack_from("<H", payload, offset)[0]
        print(f"At offset {offset}: key = '{key}', data_length = {data_length}")
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
    
    # 1. Load from file
    print("\n1. Loading and decrypting from file...")
    loaded_serialized = load_from_file("src/bin/raw_1759829184.bin")
    
    # 2. Deserialize back to objects
    print("\n2. Deserializing data...")
    loaded_data = deserialize_data(loaded_serialized)
    print_waveform_data(loaded_data)
    
    print("\n=== Example Complete ===\n")


if __name__ == "__main__":
    main()