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
import littlefs


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


def hex_to_ascii(hex_string: str, show_non_printable: bool = True) -> str:
    """
    Convert a hex string to ASCII representation.
    
    Args:
        hex_string (str): Hex string (with or without spaces/separators)
        show_non_printable (bool): Whether to show non-printable chars as dots
    
    Returns:
        str: ASCII representation of the hex data
    """
    # Clean the hex string - remove spaces, colons, hyphens
    clean_hex = ''.join(c for c in hex_string if c in '0123456789ABCDEFabcdef')
    
    # Ensure even number of characters
    if len(clean_hex) % 2 != 0:
        clean_hex = '0' + clean_hex
    
    # Convert hex pairs to ASCII
    ascii_result = ""

    # Iterate over each byte pair
    for i in range(0, len(clean_hex), 2):
        # Get the hex pair
        hex_pair = clean_hex[i:i+2]
        # Convert hex pair to byte value
        try:
            byte_value = int(hex_pair, 16)
            # Check if byte is within the printable ASCII range
            if 32 <= byte_value <= 126:  # Printable ASCII range
                ascii_result += chr(byte_value)
            # If byte is not printable, use dot
            else:
                ascii_result += '.' if show_non_printable else f'\\x{hex_pair}'
        # If hex pair is invalid, use '?'
        except ValueError:
            ascii_result += '?'
    
    # Return the final ASCII result
    return ascii_result


def bytes_to_ascii(data: bytes, show_non_printable: bool = True) -> str:
    """
    Convert bytes directly to ASCII representation.
    
    Args:
        data (bytes): Raw byte data
        show_non_printable (bool): Whether to show non-printable chars as dots
    
    Returns:
        str: ASCII representation of the byte data
    """
    # Local string to hold ASCII result
    ascii_result = ""
    
    # Convert each byte to ASCII or dot
    for byte_value in data:
        # Check if byte is printable
        if 32 <= byte_value <= 126:  # Printable ASCII range
            ascii_result += chr(byte_value)
        # If byte is not printable, use dot
        else:
            ascii_result += '.' if show_non_printable else f'\\x{byte_value:02x}'

    # Return the final ASCII result
    return ascii_result


def hex_dump_with_ascii(data: bytes, offset: int = 0, length: int = 256) -> str:
    """
    Create a hex dump with ASCII representation (like hexdump -C).
    
    Args:
        data (bytes): Raw byte data
        offset (int): Starting offset in the data
        length (int): Number of bytes to display
    
    Returns:
        str: Formatted hex dump with ASCII
    """
    # Local list to hold each line of the dump
    result = []
    # Calculate the end offset
    end_offset = min(offset + length, len(data))

    # Iterate over each 16-byte chunk
    for i in range(offset, end_offset, 16):
        # Address
        line = f"{i:08x}  "
        
        # Hex bytes
        hex_part = ""
        ascii_part = ""

        # Iterate over each byte in the chunk
        for j in range(16):
            # Check if byte is within the current chunk
            if i + j < end_offset:
                # Get the byte value
                byte_val = data[i + j]
                # Check if byte is printable
                hex_part += f"{byte_val:02x} "
                # If byte is not printable, use dot
                ascii_part += chr(byte_val) if 32 <= byte_val <= 126 else '.'
            
            # If byte is not within the current chunk, use spaces
            else:
                hex_part += "   "
                ascii_part += " "
            
            # Add extra space after 8 bytes
            if j == 7:
                hex_part += " "
        
        # Combine hex and ASCII parts
        line += hex_part + " |" + ascii_part + "|"
        result.append(line)
    
    # Join all lines into a single string
    return "\n".join(result)


def extract_files_from_littlefs(memory_data: bytes, output_dir: str = "extracted_files") -> dict:
    """
    Extract files from LittleFS filesystem in memory.
    
    Arguments:
        memory_data (bytes): Raw memory data containing LittleFS filesystem.
        output_dir (str): Directory to extract files to (default: "extracted_files").
    
    Returns:
        dict: Dictionary with file paths as keys and file contents as values.
    
    Raises:
        Exception: If LittleFS cannot be mounted or files cannot be extracted.
    """
    # Local dictionary to hold extracted files
    extracted_files = {}
    temp_file_path = None

    # Check if littlefs library is available
    try:
        # Check if littlefs library is available and get version info
        try:
            print(f"LittleFS library version: {littlefs.__version__ if hasattr(littlefs, '__version__') else 'unknown'}")
        except:
            pass
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Extracting files to: {output_dir}")
        
        # Import tempfile for temporary file handling
        import tempfile
        
        # Create a temporary file with the memory data
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(memory_data)
            temp_file_path = temp_file.name
        
        # Try different block sizes and API approaches
        mounted_fs = None
        
        # Try different block sizes
        for block_size in [4096, 512, 1024, 2048, 8192]:
            block_count = len(memory_data) // block_size
            # Skip if no complete blocks
            if block_count < 1:
                continue

            # Print current block size and count
            print(f"Trying block_size={block_size}, block_count={block_count}")
            
            # Approach 1: Use UserContext with buffer
            try:
                # Create UserContext and buffer
                ctx = littlefs.UserContext(len(memory_data))
                # Initialize buffer
                ctx.buffer[:] = memory_data
                # Create LittleFS instance
                fs = littlefs.LittleFS(context=ctx, mount=False)
                # Mount the filesystem
                fs.mount()
                # Verify the filesystem is mounted
                mounted_fs = fs
                # Print success message
                print(f"✅ Mounted with approach 1: UserContext")
                break
            # Approach 1 failed
            except Exception as e1:
                print(f"❌ Approach 1 failed: {e1}")
            
            # Approach 2: UserContext with auto-mount
            try:
                # Create UserContext and buffer
                ctx = littlefs.UserContext(len(memory_data))
                # Initialize buffer
                ctx.buffer[:] = memory_data
                # Create LittleFS instance with auto-mount
                fs = littlefs.LittleFS(context=ctx, mount=True)
                # Mount the filesystem
                mounted_fs = fs
                #  Print success message
                print(f"✅ Mounted with approach 2: UserContext auto-mount")
                break
            
            # Approach 2 failed
            except Exception as e2:
                print(f"❌ Approach 2 failed: {e2}")
            
            # Approach 3: Try with just context creation
            try:
                # Create UserContext and buffer
                ctx = littlefs.UserContext(len(memory_data))
                # Initialize buffer
                ctx.buffer[:] = memory_data
                # Create LittleFS instance
                fs = littlefs.LittleFS(context=ctx)
                # Mount the filesystem
                mounted_fs = fs  
                #  Print success message
                print(f"✅ Mounted with approach 3: context only")
                break
            # Approach 3 failed
            except Exception as e3:
                print(f"❌ Approach 3 failed: {e3}")
        
        # If no approach succeeded, raise error
        if mounted_fs is None:
            # Final fallback - try basic raw analysis
            print("All LittleFS mounting approaches failed. Attempting raw analysis...")
            return extract_files_raw_analysis(memory_data, output_dir)
         
        # If we have a mounted filesystem, proceed to extract files   
        fs = mounted_fs
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Extracting files to: {output_dir}")
        
        # List and extract all files
        def extract_recursive(path="/"):
            # List items in the current directory
            try:
                # Get list of items in directory
                items = fs.listdir(path)
                # Iterate over items
                for item in items:
                    # Construct full item path
                    item_path = f"{path.rstrip('/')}/{item}" if path != "/" else f"/{item}"

                    # Check if item is a directory or file
                    try:
                        # Check if it's a directory by trying to list it
                        try:
                            # Try to list the directory
                            fs.listdir(item_path)
                            # It's a directory
                            local_dir = os.path.join(output_dir, item_path.lstrip('/'))
                            # Create local directory
                            os.makedirs(local_dir, exist_ok=True)
                            # Recurse into directory
                            extract_recursive(item_path)

                        # If listing failed, it's a file
                        except:
                            # It's a file
                            try:
                                # Read file data
                                with fs.open(item_path, 'rb') as f:
                                    file_data = f.read()
                                
                                # Save to local filesystem
                                local_path = os.path.join(output_dir, item_path.lstrip('/'))
                                local_dir = os.path.dirname(local_path)
                                # Ensure directory exists
                                if local_dir:
                                    os.makedirs(local_dir, exist_ok=True)

                                # Write file data
                                with open(local_path, 'wb') as f:
                                    f.write(file_data)
                                
                                # Store in results
                                extracted_files[item_path] = file_data
                                print(f"Extracted: {item_path} ({len(file_data)} bytes)")

                            # If writing failed, log the error
                            except Exception as file_e:
                                print(f"Error reading file {item_path}: {file_e}")

                    # If processing failed, log the error
                    except Exception as e:
                        print(f"Error processing {item_path}: {e}")
            
            # If listing directory failed, log the error
            except Exception as e:
                print(f"Error listing directory {path}: {e}")
        
        # Start extraction from root
        extract_recursive("/")
        
        # Unmount filesystem and cleanup
        try:
            fs.unmount()
        except:
            pass
            
        # Clean up temporary file
        if temp_file_path:
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        print(f"Extraction complete. Found {len(extracted_files)} files.")
        
    # Handle any exceptions during LittleFS extraction
    except Exception as e:
        print(f"LittleFS extraction failed: {e}")
        # Make sure temp file is cleaned up even on error
        if temp_file_path:
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise
    
    # Return the extracted files dictionary
    return extracted_files


def extract_files_raw_analysis(memory_data: bytes, output_dir: str) -> dict:
    """
    Fallback function to extract files using raw memory analysis when LittleFS mounting fails.
    
    Arguments:
        memory_data (bytes): Raw memory data.
        output_dir (str): Directory to extract files to.
    
    Returns:
        dict: Dictionary with any extracted data.
    """
    # Initialize extracted files dictionary
    print("Performing raw memory analysis for file patterns...")
    extracted_files = {}
    
    # Create output directory
    try:
        # Look for common file patterns and signatures
        file_signatures = {
            b'\x89PNG\r\n\x1a\n': '.png',
            b'\xff\xd8\xff': '.jpg',
            b'GIF8': '.gif',
            b'%PDF': '.pdf',
            b'PK\x03\x04': '.zip',
            b'\x7fELF': '.elf',
            b'MZ': '.exe',
            # Protocol buffer patterns
            b'\x08\x96\x01': '.pb',
            b'\x12\x04': '.pb',
            # Common embedded patterns
            b'littlefs': '.txt',
        }
        # Variables to track found files
        found_files = 0
        offset = 0
        
        # While scanning through memory
        while offset < len(memory_data) - 16:
            # Check each signature
            for signature, extension in file_signatures.items():
                # Check if the signature matches
                if memory_data[offset:offset+len(signature)] == signature:
                    # Found a potential file
                    print(f"Found potential {extension} file at offset 0x{offset:08x}")
                    
                    # Try to extract reasonable amount of data
                    max_size = min(64 * 1024, len(memory_data) - offset)  # Max 64KB
                    file_data = memory_data[offset:offset + max_size]
                    
                    # Save the file
                    filename = f"extracted_{found_files:03d}_0x{offset:08x}{extension}"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Write to file
                    with open(filepath, 'wb') as f:
                        f.write(file_data)
                    
                    # Store in results
                    extracted_files[f"/{filename}"] = file_data
                    found_files += 1
                    
                    # Skip ahead to avoid overlapping extractions
                    offset += len(signature)
                    break
                
                # If no signature matched, move to the next byte
                else:
                    offset += 1

        # Print summary of raw analysis
        print(f"Raw analysis found {found_files} potential files")
    
    # Handle any exceptions during raw analysis    
    except Exception as e:
        print(f"Raw analysis failed: {e}")
    
    # Return any extracted files
    return extracted_files


def analyze_memory_for_littlefs(memory_data: bytes) -> dict:
    """
    Analyze memory data to detect potential LittleFS filesystem structures.
    
    Arguments:
        memory_data (bytes): Raw memory data to analyze.
    
    Returns:
        dict: Analysis results with potential filesystem information.
    """
    
    # Local dictionary to hold analysis results
    analysis = {
        "size": len(memory_data),
        "potential_filesystems": [],
        "magic_signatures": []
    }
    
    # Look for LittleFS magic signatures
    littlefs_signatures = [
        b"littlefs",
        b"\x00\x00\x00\x20",  # Common LittleFS block header
        b"\xff\xff\xff\xff",  # Erased flash pattern
    ]
    
    # Search for signatures
    for i, signature in enumerate(littlefs_signatures):
        offset = 0
        # Search for all occurrences of the signature
        while True:
            # Find the next occurrence of the signature
            pos = memory_data.find(signature, offset)
            # Break if not found
            if pos == -1:
                break
            # Record the found signature
            analysis["magic_signatures"].append({
                "signature": signature.hex(),
                "offset": pos,
                "description": ["LittleFS string", "Block header", "Erased flash"][i]
            })
            # Move offset forward to continue searching
            offset = pos + 1
    
    # Check for potential filesystem boundaries (aligned to common block sizes)
    for block_size in [512, 1024, 2048, 4096, 8192]:
        # If memory size is a multiple of block size, consider it
        if len(memory_data) % block_size == 0:
            # Record potential filesystem configuration
            analysis["potential_filesystems"].append({
                "block_size": block_size,
                "block_count": len(memory_data) // block_size,
                "alignment": "perfect"
            })
    
    # Print summary of analysis
    print(f"Memory analysis:")
    print(f"  Size: {analysis['size']} bytes")
    print(f"  Magic signatures found: {len(analysis['magic_signatures'])}")
    print(f"  Potential filesystem configs: {len(analysis['potential_filesystems'])}")
    
    # Return the analysis results
    return analysis


def read_binary_file(filename: str, max_display_bytes: int = 1024) -> bytes:
    """
    Read a binary file and display its contents in various formats.
    
    Arguments:
        filename (str): Path to the binary file to read.
        max_display_bytes (int): Maximum number of bytes to display (default: 256).
    
    Returns:
        bytes: The complete file contents as bytes.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        OSError: If there's an error reading the file.
    """
    # Read the binary file
    try:
        # Read the entire file
        with open(filename, 'rb') as f:
            data = f.read()
        
        print(f"Binary file: {filename}")
        print(f"File size: {len(data)} bytes")

        # Check if the file is empty
        if len(data) == 0:
            print("File is empty")
            return data
        
        # Display limited data for readability
        display_data = data[:max_display_bytes]
        
        # Display hex representation
        print(f"Hex representation (first {len(display_data)} bytes):")
        print(display_data.hex())
        print(f"\nASCII representation (first {len(display_data)} bytes):")
        print(hex_to_ascii(display_data.hex()))
        # Return the full data
        return data

    # Handle file reading errors
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        raise
    # Handle other I/O errors
    except OSError as e:
        print(f"Error reading file '{filename}': {e}")
        raise


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
    memory = bytearray()
    
    # Read from file the memory
    with open("src/bin/memory_dump.bin", 'rb') as f:
        memory = f.read()
    
        
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
    decrypted_serialized_data = decrypt_serialized_data(bytes(iv), proto_data_length, ciphertext)
    
    # Return the proto data length and decrypted serialized data
    return decrypted_serialized_data


def process_memory_dump_with_littlefs(filename: str, extract_dir: str = "extracted_littlefs") -> dict:
    """
    Process a memory dump file that might contain LittleFS filesystem data.
    
    Arguments:
        filename (str): Path to the memory dump file.
        extract_dir (str): Directory to extract LittleFS files to.
    
    Returns:
        dict: Results containing analysis and extracted files.
    """
    print(f"\n=== Processing Memory Dump: {filename} ===")
    
    # Read the memory dump
    memory_data = read_binary_file(filename, max_display_bytes=32)
    
    # Analyze for LittleFS structures
    print("\n--- Memory Analysis ---")
    analysis = analyze_memory_for_littlefs(memory_data)
    
    # Prepare results dictionary
    results = {
        "analysis": analysis,
        "extracted_files": {},
        "success": False
    }
    
    print("\n--- LittleFS Extraction ---")
    
    # Try to extract files from LittleFS
    try:
        # Call the extraction function
        extracted_files = extract_files_from_littlefs(memory_data, extract_dir)
        # Store extracted files in results
        results["extracted_files"] = extracted_files
        results["success"] = True
        
        # Try to process any protobuf files found
        print("\n--- Processing Extracted Protobuf Files ---")
        
        # Look for .bin, .proto, .pb files
        for file_path, file_data in extracted_files.items():
            # Only process likely protobuf files
            if file_path.endswith(('.bin', '.proto', '.pb')):
                print(f"\nAnalyzing: {file_path}")
                try:
                    # Try to parse as protobuf data directly
                    proto_data = {'waveform': file_data}
                    # Deserialize
                    deserialized = deserialize_data(proto_data)
                    # If deserialization is successful
                    if deserialized:
                        print(f"Successfully parsed protobuf data from {file_path}")
                        # Print the waveform data
                        print_waveform_data(deserialized)

                # If deserialization failed, try decryption
                except Exception as e:
                    print(f"Could not parse {file_path} as protobuf: {e}")
                    
                    # Try to decrypt if it looks like encrypted data
                    if len(file_data) > 18:  # Minimum size for IV + length + some data
                        try:
                            # Extract potential IV and data
                            iv = file_data[:16]
                            proto_length_bytes = file_data[16:18]
                            proto_length = struct.unpack('<H', proto_length_bytes)[0]
                            encrypted_data = file_data[18:]

                            # Attempt decryption
                            print(f"Attempting decryption of {file_path}...")
                            # Call the decryption function
                            decrypted = decrypt_serialized_data(iv, proto_length, encrypted_data)
                            # If decryption successful, try to deserialize
                            if decrypted:
                                # Deserialize
                                deserialized = deserialize_data(decrypted)
                                # If deserialization is successful
                                if deserialized:
                                    print(f"Successfully decrypted and parsed {file_path}")
                                    print_waveform_data(deserialized)

                        # If decryption failed, log the error
                        except Exception as decrypt_e:
                            print(f"Decryption failed for {file_path}: {decrypt_e}")
    
    # Catch any unexpected errors during extraction    
    except Exception as e:
        print(f"LittleFS extraction failed: {e}")
        results["error"] = str(e)
    
    return results


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
    
    # 3. Optional: Try LittleFS extraction if a memory dump file exists
    memory_dump_files = [
        "src/bin/memory_dump.bin",
        "src/bin/flash_dump.bin", 
        "src/bin/littlefs_dump.bin"
    ]
    
    # Check for any existing memory dump files
    for dump_file in memory_dump_files:
        # Check if file exists
        if os.path.exists(dump_file):
            print(f"\n3. Found memory dump: {dump_file}")
            # Attempt to process the memory dump
            try:
                # Call the extraction function
                results = process_memory_dump_with_littlefs(dump_file)
                # If successful, print summary
                if results["success"]:
                    print(f"Successfully extracted {len(results['extracted_files'])} files from LittleFS")
                # If extraction failed, print error message
                else:
                    print("LittleFS extraction failed or no files found")
            
            # Catch any exceptions during processing
            except Exception as e:
                print(f"Error processing memory dump: {e}")
            break
    
    # If no memory dump files were found, print a message
    else:
        print("\n3. No memory dump files found for LittleFS extraction")
        print("   Place memory dumps in src/bin/ with names like:")
        print("   - memory_dump.bin")
        print("   - flash_dump.bin") 
        print("   - littlefs_dump.bin")
    
    # Finalize
    print("\n=== Example Complete ===\n")


def test_littlefs_extraction(memory_file: str) -> None:
    """
    Test function to extract files from a LittleFS memory dump.
    
    Arguments:
        memory_file (str): Path to the memory dump file containing LittleFS data.
    
    Returns:
        None
    """
    print(f"\n=== LittleFS Extraction ===")
    print(f"Processing: {memory_file}")
    
    # Check if file exists
    if not os.path.exists(memory_file):
        print(f"Error: File {memory_file} not found")
        print("Usage: test_littlefs_extraction('path/to/memory_dump.bin')")
        return
    
    # Process the memory dump
    try:
        # Call the extraction function
        results = process_memory_dump_with_littlefs(memory_file)
        
        # If successful, print extracted files
        if results["success"]:
            print(f"\n✅ Success! Extracted {len(results['extracted_files'])} files")
            print("Extracted files:")
            for file_path in results["extracted_files"].keys():
                print(f"  - {file_path}")

        # If extraction failed, print error message
        else:
            print("❌ LittleFS extraction failed")
            if "error" in results:
                print(f"Error: {results['error']}")
    
    # Catch any exceptions during extraction            
    except Exception as e:
        print(f"❌ Exception during extraction: {e}")

    # Finalize extraction process
    print("=== Test Complete ===\n")


if __name__ == "__main__":
    main()