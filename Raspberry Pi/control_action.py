import RPi.GPIO as GPIO
import time

def blink_light():
    #blink the light
    GPIO.setmode(GPIO.BOARD)
    # set up GPIO output channel
    GPIO.setup(11, GPIO.OUT)
    GPIO.output(11, GPIO.HIGH)
    # wait half a second
    time.sleep(1)
    # set port/pin value to 1/GPIO.HIGH/True
    GPIO.output(11, GPIO.LOW)
    # wait half a second
    time.sleep(1)
    # cleanup GPIO settings
    GPIO.cleanup()

def lights_off():
    #turn off the light
    GPIO.setmode(GPIO.BOARD)
    # set up GPIO output channel
    GPIO.setup(11, GPIO.OUT)
    # set port/pin value to 0/GPIO.LOW/False
    GPIO.output(11, GPIO.LOW)
    # cleanup GPIO settings
    GPIO.cleanup()
    
def lights_on():
    #turn on the light
    GPIO.setmode(GPIO.BOARD)
    # set up GPIO output channel
    GPIO.setup(11, GPIO.OUT)
    # set port/pin value to 1/GPIO.HIGH/True
    GPIO.output(11, GPIO.HIGH)
    # cleanup GPIO settings
    GPIO.cleanup()
    
def servo_motor_control():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(12, GPIO.OUT)
    # set GPIO 12 as PWM output, with 50Hz frequency
    p = GPIO.PWM(12, 50)
    # start PWM running, with value of 0 (pulse off)
    p.start(7.5)
    try:
        while True:
            p.ChangeDutyCycle(7.5)  # turn towards 90 degree
            time.sleep(1) # sleep 1 second
            p.ChangeDutyCycle(2.5)  # turn towards 0 degree
            time.sleep(1) # sleep 1 second
            p.ChangeDutyCycle(12.5) # turn towards 180 degree
            time.sleep(1) # sleep 1 second 
    except KeyboardInterrupt:
        p.stop()
        GPIO.cleanup()