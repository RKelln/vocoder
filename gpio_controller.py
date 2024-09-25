import RPi.GPIO as GPIO
import threading
import time
import queue

import logging
logger = logging.getLogger(__name__)


class GPIOController:
    def __init__(self, pin:int, frequency:float=1000, input_range:tuple[int]=(0, 255)):
        self.pin = pin
        self.frequency = frequency
        self.input_range = input_range
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, self.frequency)
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.run)
        self.running = False

    def run(self):
        while self.running:
            try:
                duty_cycle = self.queue.get(timeout=1)
                # map input range to duty cycle (0-100)
                duty_cycle = (duty_cycle - self.input_range[0]) / (self.input_range[1] - self.input_range[0]) * 100
                self.pwm.ChangeDutyCycle(duty_cycle)
            except queue.Empty:
                continue

    def update_pwm(self, value):
        if self.running and 0 <= value:
            self.queue.put(value)

    def start(self):
        self.running = True
        self.pwm.start(0)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.pwm.stop()
        GPIO.cleanup()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class GPIOManager:
    def __init__(self, pins:list[int], frequency:float=1000, input_range:tuple[int]=(0, 255)):
        GPIO.setmode(GPIO.BCM)
        self.controllers = {pin: GPIOController(pin, frequency, input_range) for pin in pins}

    def update_pwm(self, pin, value):
        if pin in self.controllers:
            self.controllers[pin].update_pwm(value)

    def update(self, values):
        if len(values) != len(self.controllers):
            raise ValueError("Number of values must match number of pins")
        for pin, value in zip(self.controllers.keys(), values):
            self.update_pwm(pin, value)

    def start_all(self):
        for controller in self.controllers.values():
            controller.start()

    def stop_all(self):
        for controller in self.controllers.values():
            controller.stop()
        GPIO.cleanup()

    def __enter__(self):
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_all()


# Usage in audio_processor.py
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GPIO controller")
    parser.add_argument("--info", action="store_true", help="Print available GPIO pins")
    args = parser.parse_args()

    # if --info is passed, print the available pins
    if args.info:
        print("Available GPIO pins:")
        from gpiozero import Device
        from gpiozero.pins.pigpio import PiGPIOFactory

        factory = PiGPIOFactory()
        gpio_pins = Device.pin_class.pins

        print(f"Total GPIO pins available: {len(gpio_pins)}")
        print(f"GPIO pins: {gpio_pins}")

        exit(0)

    import random   
    gpio_pins = [18, 23, 24]  # Example GPIO pins
    input_range = (0, 255)
    gpio_manager = GPIOManager(pins=gpio_pins, input_range=input_range)
    try:
        with gpio_manager:
            while True:
                # Update PWM values for each pin
                random_values = [random.randint(*input_range) for _ in gpio_pins]
                gpio_manager.update(random_values)
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("Exiting...")