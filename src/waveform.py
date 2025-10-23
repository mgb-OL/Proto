from datetime import datetime

class PPG:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate
        self.ir = []
        self.red = []
        self.green = []
    
    def extract_samples(self, samples: list) -> None:
        for sample in samples:
            self.ir.append(sample.ir)
            self.red.append(sample.red)
            self.green.append(sample.green)

class IMU:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate
        self.values = []

    def extract_samples(self, values: list) -> None:
        self.values.append([value.norm for value in values])

class ABP:
    def __init__(self, systolic: int, diastolic: int) -> None:
        self.systolic = systolic
        self.diastolic = diastolic


class Waveform:

    def __init__(self) -> None:
        self.file_name = None
        self.device_SN = None
        self.userID = None
        self.start_time = None
        self.end_time = None
        self.pilot = 'PILOT: VALL D\'HEBRON'
        self.epoch = None
        self.PPG = None
        self.IMU = None
        self.hr = None
        self.spo2 = None
        self.ABP = None
        self.skin_temperature = None

    def set_metadata(self, file_name: str, device_SN: str, userID: str, start_time: datetime, end_time: datetime) -> None:
        """Set file-related information."""
        self.file_name = file_name
        self.device_SN = device_SN
        self.userID = userID
        self.start_time = start_time
        self.end_time = end_time
        
    def extract_data(self, decoded_data: dict) -> None:
        """Extract data from decoded protobuf waveform."""
        # Get the waveform protobuf object
        wf = decoded_data['waveform']
        
        # Extract individual fields from the protobuf object
        self.epoch = datetime.fromtimestamp(wf.epoch)
        
        # PPG data
        self.PPG = PPG(
            sampling_rate=wf.ppg.samplingRate,
        )
        self.PPG.extract_samples(wf.ppg.ppg_samples)

        # IMU data
        self.IMU = IMU(
            sampling_rate=wf.imu.samplingRate
        )
        self.IMU.extract_samples(wf.imu.acc)

        # Vital signs
        self.hr = wf.heartRate.value
        self.spo2 = wf.oxigenSaturation.value

        # Blood pressure
        self.ABP = ABP(
            systolic=wf.arterialBloodPressure.systolic,
            diastolic=wf.arterialBloodPressure.diastolic
        )
        
        # Temperature
        self.skin_temperature = wf.skinTemperature.value
