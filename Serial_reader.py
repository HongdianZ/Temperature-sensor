import serial
import h5py
import datetime

START_FLAG = 0xAB
END_FLAG = 0xCD
ESCAPE_BYTE = 0x7D
ESCAPE_MASK = 0x20

def read_from_serial(port, baudrate, output_file):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Connected to {port} at {baudrate} baud.")
    except serial.SerialException as e:
        print(f"Could not open port {port}: {e}")
        return

    adc_count = 0
    receiving_data = False
    received_bytes = bytearray()
    adc_frame = []

    try:
        # Read and print the initial "Hello World" message
        hello_message = ser.readline().decode('utf-8').strip()
        print(f"Received initial message: {hello_message}")

        with h5py.File(output_file, 'w') as h5file:
            while True:
                if ser.in_waiting > 0:
                    byte = ser.read(1)[0]

                    if not receiving_data:
                        if byte == START_FLAG:  # Start flag detected
                            receiving_data = True
                            received_bytes.clear()
                    else:
                        if byte == END_FLAG:  # End flag detected
                            # Process the received frame
                            if len(received_bytes) == 4 * 4 * 2:
                                adc_frame = decode_adc_frame(received_bytes)
                                timestamp = datetime.datetime.now().isoformat().encode('utf-8')
                                frame_group = h5file.create_group(f'frame_{adc_count}')
                                frame_group.create_dataset('timestamp', data=timestamp)
                                frame_group.create_dataset('adc_values', data=adc_frame)
                                adc_count += 1
                                print(f"Total frames received: {adc_count}")
                            receiving_data = False
                        elif byte == ESCAPE_BYTE:  # Escape byte detected
                            next_byte = ser.read(1)[0]
                            received_bytes.append(next_byte ^ ESCAPE_MASK)
                        else:
                            received_bytes.append(byte)
    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        ser.close()
        print(f"Disconnected from {port}.")

def decode_adc_frame(data):
    adc_values = []
    for i in range(0, len(data), 2):
        high_byte = data[i]
        low_byte = data[i + 1]
        adc_value = (high_byte << 8) | low_byte
        adc_values.append(adc_value)
    return adc_values

if __name__ == "__main__":
    read_from_serial('COM3', 2000000, 'adc_data.h5')
