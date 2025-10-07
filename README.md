# PROTO

A Python project for prototyping and development with Protocol Buffers, AES encryption, and LittleFS extraction capabilities.

## Features

- **Protocol Buffers**: Serialization and deserialization of waveform data (PPG, IMU, health metrics)
- **AES Encryption**: CBC mode encryption/decryption with PKCS#7 padding
- **File Format Support**: Both custom key/data format and raw protobuf data
- **LittleFS Integration**: Extract files from embedded device memory dumps
- **Binary Analysis**: Comprehensive binary file reading and analysis tools

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Protocol Buffer Operations

Run the main script:

```bash
python src/proto_example.py
```

### LittleFS Memory Extraction

For extracting files from embedded device memory dumps:

```bash
# Run the LittleFS example
python example_littlefs_usage.py

# Or use directly in Python
from src.proto_example import test_littlefs_extraction
test_littlefs_extraction('path/to/memory_dump.bin')
```

### Available Functions

#### Protocol Buffer & Encryption

- `create_waveform_data()` - Generate sample waveform data
- `serialize_data(data)` - Serialize protobuf messages
- `encrypt_serialized_data(data)` - AES encrypt serialized data
- `decrypt_serialized_data(iv, length, ciphertext)` - AES decrypt data
- `deserialize_data(data)` - Deserialize protobuf messages

#### LittleFS Operations

- `extract_files_from_littlefs(memory_data)` - Extract files from LittleFS
- `analyze_memory_for_littlefs(memory_data)` - Analyze memory for LittleFS structures
- `process_memory_dump_with_littlefs(filename)` - Full memory dump processing
- `test_littlefs_extraction(filename)` - Simple test function

#### Utility Functions

- `read_binary_file(filename)` - Read and analyze binary files
- `invert_byte_array(bytes)` - Reverse byte order
- `get_sub_byte_array(bytes, start, length)` - Extract byte array segment

## File Formats Supported

### Input Files

- **Encrypted Proto Files**: Custom format with IV + length headers + AES-CBC encrypted protobuf data
- **Raw Proto Files**: Direct protobuf serialized data
- **Memory Dumps**: Raw memory/flash dumps containing LittleFS filesystem
- **Binary Files**: Any binary file for analysis

### Output

- **Extracted Files**: Files extracted from LittleFS to local filesystem
- **Decrypted Data**: Protobuf messages with waveform/health data
- **Analysis Reports**: Memory structure analysis and filesystem information

## Memory Dump Processing

Place memory dump files in `src/bin/` with names like:

- `memory_dump.bin`
- `flash_dump.bin`
- `littlefs_dump.bin`

The system will automatically detect and process LittleFS filesystems with various block sizes (512, 1024, 2048, 4096, 8192 bytes).

## Usage

Run the main script:

```bash
python src/main.py
```

## Development

### Project Structure

```
PROTO/
├── src/                    # Source code
│   ├── __init__.py
│   └── main.py
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_main.py
├── docs/                   # Documentation
├── .github/                # GitHub configuration
│   └── copilot-instructions.md
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py
```

### Running Tests

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License.
