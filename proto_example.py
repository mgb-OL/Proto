#!/usr/bin/env python3
"""
Example of how to read and use Protocol Buffer data.
"""

# Import the generated protobuf classes
import importlib.util, sys, os, struct
import tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Sequence, Tuple
from Crypto.Cipher import AES
from google.protobuf.message import DecodeError
from littlefs import LittleFS, LittleFSError, UserContext




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
        # Resolve candidate locations so relative paths work regardless of cwd.
        module_dir = Path(__file__).resolve().parent
        project_root = module_dir.parent
        raw_path = Path(key_file_path).expanduser()
        candidate_paths = []

        if raw_path.is_file():
            candidate_paths.append(raw_path)
        else:
            for base in (module_dir, project_root):
                candidate = (base / key_file_path).resolve()
                if candidate.is_file():
                    candidate_paths.append(candidate)
                    break

        for candidate in candidate_paths:
            try:
                file_bytes = candidate.read_bytes()
            except OSError:
                continue

            if len(file_bytes) in {16, 24, 32}:
                return file_bytes

    raise RuntimeError(
        "Unable to load AES key. Set PROTO_ENCRYPTION_KEY or PROTO_ENCRYPTION_KEY_FILE to a valid 16/24/32-byte key."
    )

# LOAD THE ENCRYPTION KEY
ENCRYPTION_KEY_BYTES = _load_encryption_key()



# LOAD THE PROTOBUF MODULE
# Dynamically load the raw_data protobuf module
spec = importlib.util.spec_from_file_location("raw_data_pb2", os.path.join(os.path.dirname(__file__), "src","raw_data_pb2.py"))
# Load the raw_data protobuf module
raw_data_pb2 = importlib.util.module_from_spec(spec)
# Register the module in sys.modules
sys.modules["raw_data_pb2"] = raw_data_pb2
# Execute the module
spec.loader.exec_module(raw_data_pb2)


DEFAULT_LITTLEFS_BLOCK_SIZES: tuple[int, ...] = (4096, 2048, 1024, 512, 256, 128)
DEFAULT_LITTLEFS_LOOKAHEAD = 128
EXTRACTION_CHUNK_SIZE = 64 * 1024



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
        # Unpack the serialized data
        serialized_data = unpack_serialized_data(payload, proto_data_length)
        # Return the unpacked serialized data
        return serialized_data
    
    # If unpacking fails, treat as raw protobuf data
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
    # Validate start index
    if start < 0:
        raise ValueError("Start index cannot be negative")
    
    # Validate start within bounds
    if start >= len(byte_array):
        raise ValueError(f"Start index {start} is out of bounds for array of length {len(byte_array)}")
    
    # If length is None, extract to the end
    if length is None:
        return byte_array[start:]
    
    # Validate length
    if length < 0:
        raise ValueError("Length cannot be negative")
    
    # Validate end within bounds
    end = start + length
    # If end is None, extract to the end
    if end > len(byte_array):
        raise ValueError(f"End index {end} would exceed array bounds of {len(byte_array)}")
    
    # Return the sub-array
    return byte_array[start:end]









def load_from_file(filename: str) -> dict:
    """Load serialized protobuf data from disk, handling encrypted and raw layouts."""
    path = Path(filename)
    if not path.is_file():
        path = _resolve_binary_path(filename)

    file_bytes = path.read_bytes()

    if len(file_bytes) >= 18:
        iv = file_bytes[:16]
        proto_length = struct.unpack('<H', file_bytes[16:18])[0]
        ciphertext = file_bytes[18:]

        ciphertext_has_blocks = len(ciphertext) % 16 == 0 and len(ciphertext) > 0
        if ciphertext_has_blocks and 0 < proto_length <= len(ciphertext):
            try:
                return decrypt_serialized_data(iv, proto_length, ciphertext)
            except ValueError:
                pass

    return {'waveform': file_bytes}





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


def _resolve_binary_path(filename: str) -> Path:
    """Resolve the absolute path to a binary resource inside the project."""
    candidate = Path(filename)
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent

    variants = [candidate]
    if len(candidate.parts) > 1:
        variants.append(Path(candidate.name))

    bases: List[Path | None] = [
        None,
        Path.cwd(),
        module_dir,
        module_dir / "bin",
        project_root,
        project_root / "src",
        project_root / "src" / "bin",
    ]

    seen = set()
    for base in bases:
        for variant in variants:
            if variant.is_absolute():
                target = variant
            elif base is None:
                target = variant
            else:
                target = base / variant

            try:
                resolved = target.resolve()
            except OSError:
                continue

            if resolved in seen:
                continue
            seen.add(resolved)

            if resolved.is_file():
                return resolved

    raise FileNotFoundError(f"Unable to locate binary file '{filename}' relative to the project")


def _mount_littlefs_from_bytes(data: bytes, block_size: int, lookahead_size: int) -> Tuple[LittleFS, UserContext, int]:
    """Create and mount a LittleFS instance backed by in-memory bytes."""
    if block_size <= 0:
        raise ValueError("Block size must be a positive integer")

    block_count = len(data) // block_size
    if block_count == 0:
        raise ValueError("Block size is larger than the available data length")

    context = UserContext(len(data))
    context.buffer[:] = data

    fs = LittleFS(
        context=context,
        block_size=block_size,
        block_count=block_count,
        read_size=block_size,
        prog_size=block_size,
        cache_size=block_size,
        lookahead_size=lookahead_size,
        mount=False,
    )
    fs.mount()
    return fs, context, block_count


def _extract_littlefs_files(fs: LittleFS, output_dir: Path) -> List[Dict[str, Any]]:
    """Extract all files from a mounted LittleFS instance into an output directory."""
    extracted: List[Dict[str, Any]] = []
    chunk_size = EXTRACTION_CHUNK_SIZE

    for root, _, files in fs.walk("/"):
        for filename in files:
            posix_path = PurePosixPath(root) / filename
            relative_path = posix_path.relative_to("/")
            local_path = output_dir.joinpath(*relative_path.parts)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            lfs_path = "/" + relative_path.as_posix()

            file_size = 0
            with fs.open(lfs_path, "rb") as src, open(local_path, "wb") as dst:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    file_size += len(chunk)

            extracted.append(
                {
                    "littlefs_path": lfs_path,
                    "output_path": str(local_path),
                    "size": file_size,
                }
            )

    extracted.sort(key=lambda entry: entry["littlefs_path"])
    return extracted


def read_binary_file(filename: str) -> bytes:
    """Read the content of a binary file from common project locations."""
    path = _resolve_binary_path(filename)
    return path.read_bytes()


def analyze_memory_for_littlefs(
    data: bytes,
    block_sizes: Sequence[int] = DEFAULT_LITTLEFS_BLOCK_SIZES,
    lookahead_size: int = DEFAULT_LITTLEFS_LOOKAHEAD,
) -> Dict[str, Any]:
    """Attempt to discover LittleFS configurations present in raw memory bytes."""
    successful_mounts: List[Dict[str, Any]] = []

    for block_size in block_sizes:
        fs: LittleFS | None = None
        try:
            fs, _, block_count = _mount_littlefs_from_bytes(data, block_size, lookahead_size)
        except (LittleFSError, ValueError):
            continue

        try:
            tree_overview: List[Dict[str, Any]] = []
            for root, dirs, files in fs.walk("/"):
                tree_overview.append(
                    {
                        "path": root,
                        "directories": list(dirs),
                        "files": list(files),
                    }
                )

            successful_mounts.append(
                {
                    "block_size": block_size,
                    "block_count": block_count,
                    "read_size": block_size,
                    "prog_size": block_size,
                    "cache_size": block_size,
                    "lookahead_size": lookahead_size,
                    "root_entries": fs.listdir("/"),
                    "tree": tree_overview,
                }
            )
        finally:
            if fs is not None:
                try:
                    fs.unmount()
                except LittleFSError:
                    pass

    return {
        "data_length": len(data),
        "attempted_block_sizes": list(block_sizes),
        "successful_mounts": successful_mounts,
    }


def process_memory_dump_with_littlefs(
    filename: str,
    output_dir: str | Path | None = None,
    *,
    block_size: int | None = None,
    block_size_candidates: Sequence[int] = DEFAULT_LITTLEFS_BLOCK_SIZES,
    lookahead_size: int = DEFAULT_LITTLEFS_LOOKAHEAD,
) -> Dict[str, Any]:
    """Extract all LittleFS files contained in a raw memory dump."""
    source_path = _resolve_binary_path(filename)
    data = source_path.read_bytes()

    chosen_block_size = block_size
    analysis: Dict[str, Any] | None = None

    if chosen_block_size is None:
        analysis = analyze_memory_for_littlefs(data, block_size_candidates, lookahead_size)
        if not analysis["successful_mounts"]:
            raise RuntimeError("Unable to locate a LittleFS filesystem in the provided dump")
        chosen_block_size = analysis["successful_mounts"][0]["block_size"]

    fs: LittleFS | None = None
    _context: UserContext | None = None
    extracted_files: List[Dict[str, Any]] = []

    try:
        fs, _context, _ = _mount_littlefs_from_bytes(data, chosen_block_size, lookahead_size)

        if output_dir is None:
            default_name = f"{source_path.stem}_extracted"
            output_path = source_path.parent / default_name
        else:
            output_path = Path(output_dir)

        output_path = output_path.resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        extracted_files = _extract_littlefs_files(fs, output_path)
    finally:
        if fs is not None:
            try:
                fs.unmount()
            except LittleFSError:
                pass

    result: Dict[str, Any] = {
        "source": str(source_path),
        "output_dir": str(output_path),
        "config": {
            "block_size": chosen_block_size,
            "block_count": len(data) // chosen_block_size,
            "lookahead_size": lookahead_size,
            "read_size": chosen_block_size,
            "prog_size": chosen_block_size,
            "cache_size": chosen_block_size,
        },
        "extracted_files": extracted_files,
    }

    if analysis is not None:
        result["analysis"] = analysis

    return result


def test_littlefs_extraction(
    filename: str = "src/bin/memory_dump.bin",
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience helper to run the extraction process and print a summary."""
    result = process_memory_dump_with_littlefs(filename, output_dir=output_dir, **kwargs)
    file_count = len(result["extracted_files"])
    print(f"Extracted {file_count} file(s) to {result['output_dir']}")
    return result


def select_memory_dump_file():
    """Open a file dialog to select a memory dump file."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    file_path = filedialog.askopenfilename(
        title="Select Memory Dump File",
        initialdir=Path.cwd() / "src" / "bin",  # Start in your bin directory
        filetypes=[
            ("Binary files", "*.bin"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        return Path(file_path)
    return None


def main() -> None:
    """
    Main function demonstrating proto file usage.
    
    Arguments:
        None
        
    Returns:
        None
    """
    # Initialize index
    i = 1
    
    # 1. Extract memory filesystem contents
    print(f"\n{i}. Select the memory dump file to process using file dialog...")
    # Increment index
    i += 1
    
    # Use file dialog to select memory dump file
    memory_dump_path = select_memory_dump_file()
    
    # If no file selected, exit
    if memory_dump_path:
        # Print selected file
        print(f"\nProcessing memory file: {memory_dump_path}\n")

        # Print instruction to enter device SN
        print(f"\n{i}. Insert the device serial number (SN) using the input dialog...")
        i += 1
        
        # Device serial number input to store extracted files
        device_SN = simpledialog.askstring("SN", "Enter the device serial number (SN):")

        if not device_SN:
            print("No serial number provided. Exiting.")
            return
        
        # Print the device SN being used
        print(f"\nUsing device SN: {device_SN}\n")
        
        # Perform extraction
        extraction_result = process_memory_dump_with_littlefs(memory_dump_path, output_dir=memory_dump_path.parent / device_SN)
        extracted_files = extraction_result["extracted_files"]
        
        
        # Identify candidate binary files for decryption
        preferred_candidates = []
        fallback_candidates = []
        raw_files_found = 0
        meas_files_found = 0

        # For each extracted file, check if it's a candidate
        for entry in extracted_files:
            # Lowercase filename for comparison
            filename = Path(entry["output_path"]).name.lower()
            # Skip measurement metadata files
            if filename.startswith("meas"):
                meas_files_found += 1
                continue
            # Consider only .bin files
            if entry["output_path"].lower().endswith(".bin"):
                # Prefer files in /data/raw/ directory
                if "/data/raw/" in entry["littlefs_path"].lower():
                    raw_files_found += 1
                    preferred_candidates.append(entry)
                # Otherwise, add to fallback list
                else:
                    fallback_candidates.append(entry)
        
        # Print extraction summary
        print(f"\n{i}. Extracted {len(extracted_files)} file(s) from binary image into {extraction_result['output_dir']}")
        print(f"\nFound {raw_files_found} raw file(s) and {meas_files_found} meas file(s) during extraction.")
        i += 1
        
        # Combine preferred and fallback candidates
        candidates = preferred_candidates + fallback_candidates

        # Initialize variables for selected file
        decoded_data: Dict[str, Any] | None = None

        # Try to decrypt each candidate file
        for entry in candidates:
            # Get the candidate file path
            candidate_path = entry["output_path"]
            # Print attempt message
            print(f"\n{i}.1 Attempting to decrypt {entry['littlefs_path'].split('/')[-1]} \n")
            
            # Attempt to load and decrypt the file
            try:
                # Load the serialized data from the candidate file
                candidate_serialized = load_from_file(candidate_path)
                # Deserialize the data
                decoded_data = deserialize_data(candidate_serialized)
            # If deserialization fails, skip the file and show error
            except (OSError, ValueError, RuntimeError, DecodeError) as exc:
                print(f"  Skipping file due to error: {exc}")
                continue
            
            # 4. Load from the selected file
            print(f"\n\n{i}.2 Loading and decrypting from file...")
            print(f"\nUsing extracted file: {candidate_path.split('\\')[-1]}\n")

            # 5. Deserialize back to objects
            print(f"\n{i}.3 Saving data")
            print_waveform_data(decoded_data)

            # Finalize extraction process
            print(f"\nExtraction from file: {candidate_path.split('\\')[-1]} completed \n")
            i += 1

        print(f"\n=== Extraction process completed ===\n")
    
    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    main()