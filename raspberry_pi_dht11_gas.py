#!/usr/bin/env python3
"""
Raspberry Pi: DHT11 + I2C LCD + gas sensor + buzzer

Wiring:
- DHT11 data pin -> GPIO 4 (pin 7)
- DHT11 VCC -> 5V or 3.3V
- DHT11 GND -> GND
- MQ-2 gas sensor digital output -> GPIO 17 (pin 11)
- Buzzer positive -> GPIO 18 (pin 12), negative -> GND
- I2C LCD SDA -> SDA (GPIO 2, pin 3)
- I2C LCD SCL -> SCL (GPIO 3, pin 5)
- I2C LCD VCC -> 5V, GND -> GND

Install:
  sudo apt update
  sudo apt install -y python3-pip python3-smbus i2c-tools
  python3 -m pip install RPi.GPIO Adafruit_DHT smbus2

Run:
  python3 raspberry_pi_dht11_gas.py
"""

import socket
import time

try:
    import Adafruit_DHT
except ImportError:
    raise ImportError("Install Adafruit_DHT: python3 -m pip install Adafruit_DHT")

try:
    import RPi.GPIO as GPIO
except ImportError:
    raise ImportError("Install RPi.GPIO: python3 -m pip install RPi.GPIO")

try:
    from smbus2 import SMBus
except ImportError:
    raise ImportError("Install smbus2: python3 -m pip install smbus2")

# GPIO pins
DHT_PIN = 4
GAS_PIN = 17
BUZZER_PIN = 18
DHT_SENSOR = Adafruit_DHT.DHT11
I2C_ADDR = 0x27
LCD_WIDTH = 16

# LCD constants
LCD_CHR = 1
LCD_CMD = 0
LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0
LCD_BACKLIGHT = 0x08
ENABLE = 0b00000100

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "No IP"
    finally:
        s.close()
    return ip


def lcd_toggle_enable(bus, bits):
    bus.write_byte(I2C_ADDR, bits | ENABLE | LCD_BACKLIGHT)
    time.sleep(E_PULSE)
    bus.write_byte(I2C_ADDR, ((bits & ~ENABLE) | LCD_BACKLIGHT))
    time.sleep(E_DELAY)


def lcd_byte(bus, bits, mode):
    high_bits = mode | (bits & 0xF0) | LCD_BACKLIGHT
    low_bits = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
    bus.write_byte(I2C_ADDR, high_bits)
    lcd_toggle_enable(bus, high_bits)
    bus.write_byte(I2C_ADDR, low_bits)
    lcd_toggle_enable(bus, low_bits)


def lcd_string(bus, message, line):
    message = message.ljust(LCD_WIDTH, " ")
    lcd_byte(bus, line, LCD_CMD)
    for char in message:
        lcd_byte(bus, ord(char), LCD_CHR)


def lcd_init(bus):
    lcd_byte(bus, 0x33, LCD_CMD)
    lcd_byte(bus, 0x32, LCD_CMD)
    lcd_byte(bus, 0x28, LCD_CMD)
    lcd_byte(bus, 0x0C, LCD_CMD)
    lcd_byte(bus, 0x06, LCD_CMD)
    lcd_byte(bus, 0x01, LCD_CMD)
    time.sleep(E_DELAY)


def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GAS_PIN, GPIO.IN)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(BUZZER_PIN, GPIO.LOW)


if __name__ == "__main__":
    try:
        setup()
        with SMBus(1) as bus:
            lcd_init(bus)
            last_display = 0
            display_stage = 0

            while True:
                humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
                gas_high = GPIO.input(GAS_PIN) == GPIO.HIGH
                ip_addr = get_ip_address()

                if gas_high:
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                else:
                    GPIO.output(BUZZER_PIN, GPIO.LOW)

                current_time = time.time()
                if current_time - last_display > 5:
                    last_display = current_time
                    display_stage = (display_stage + 1) % 2

                if display_stage == 0:
                    if temperature is not None and humidity is not None:
                        line1 = f"Temp: {temperature:0.1f}C"
                        line2 = f"Hum: {humidity:0.1f}%"
                    else:
                        line1 = "Temp/Hum: error"
                        line2 = "Retrying..."
                else:
                    line1 = f"IP: {ip_addr[:16]}"
                    line2 = "Gas: HIGH" if gas_high else "Gas: OK "

                lcd_string(bus, line1, LCD_LINE_1)
                lcd_string(bus, line2, LCD_LINE_2)

                time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        GPIO.cleanup()
