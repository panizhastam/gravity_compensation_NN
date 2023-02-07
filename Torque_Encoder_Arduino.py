import sys
import time
import torch

from telemetrix import telemetrix

"""
Monitor a digital input pin
"""

"""
Setup a pin for digital input and monitor its changes
"""

# Set up a pin for analog input and monitor its changes


# Callback data indices
CB_PIN_MODE = 0
CB_PIN = 1
CB_VALUE = 2
CB_TIME = 3


# Encoder PINs
DATA_PIN_Elbow = 2  
STATE_PIN_Elbow = 3

DATA_PIN_ShFE = 19  
STATE_PIN_ShFE = 18

DATA_PIN_ShAA = 20 
STATE_PIN_ShAA = 21


# Torque control PINs
PRESSURE_PIN_ELBOW = 6
DRIVER_PIN1_ELBOW = 17
DRIVER_PIN2_ELBOW = 5

PRESSURE_PIN_SHFE = 7
DRIVER_PIN1_SHFE = 44
DRIVER_PIN2_SHFE = 45

PRESSURE_PIN_SHAA = 8
DRIVER_PIN1_SHAA = 52
DRIVER_PIN2_SHAA = 53

################################################################################################################################################
# contribution of Ismail
# best code ever
# please dont change


prev_pin = {'Elbow': [0,1], 
            'ShFE': [0,1],
            'ShAA': [0,1]}
count ={'Elbow': 0, 
            'ShFE': 0,
            'ShAA': 0}

def compute_angle(pin_number,pin_value,joint):
    if joint == 'Elbow':        
        if pin_number == DATA_PIN_Elbow:
            if prev_pin[joint][0] != prev_pin[joint][1]:
                if prev_pin[joint][1] != pin_value:
                    count[joint] += 1
                else:
                    count[joint] -= 1
    if joint == 'ShFE':        
        if pin_number == DATA_PIN_ShFE:
            if prev_pin[joint][0] != prev_pin[joint][1]:
                if prev_pin[joint][1] == pin_value:
                    count[joint] += 1
                else:
                    count[joint] -= 1
    if joint == 'ShAA':        
        if pin_number == DATA_PIN_ShAA:
            if prev_pin[joint][0] != prev_pin[joint][1]:
                if prev_pin[joint][1] == pin_value:
                    count[joint] += 1
                else:
                    count[joint] -= 1    
    s =""
    for x in count:
        s+= str(x) + ': '+ str(count[x]*360/1024) + "   "
        
    s+="\n"
    print(s)
    prev_pin[joint][0] = pin_value

def the_callback_elbow(data):
    """
    A callback function to report data changes.
    This will print the pin number, its reported value and
    the date and time when the change occurred

    :param data: [pin, current reported value, pin_mode, timestamp]
    """
    date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[CB_TIME]))
    # print(f'Pin Mode: {data[CB_PIN_MODE]} Pin: {data[CB_PIN]} Value: {data[CB_VALUE]} Time Stamp: {date}')
    compute_angle(data[CB_PIN],data[CB_VALUE],'Elbow')

def the_callback_shfe(data):
    """
    A callback function to report data changes.
    This will print the pin number, its reported value and
    the date and time when the change occurred

    :param data: [pin, current reported value, pin_mode, timestamp]
    """
    date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[CB_TIME]))
    # print(f'Pin Mode: {data[CB_PIN_MODE]} Pin: {data[CB_PIN]} Value: {data[CB_VALUE]} Time Stamp: {date}')
    compute_angle(data[CB_PIN],data[CB_VALUE],'ShFE')

def the_callback_shaa(data):
    """
    A callback function to report data changes.
    This will print the pin number, its reported value and
    the date and time when the change occurred

    :param data: [pin, current reported value, pin_mode, timestamp]
    """
    date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[CB_TIME]))
    # print(f'Pin Mode: {data[CB_PIN_MODE]} Pin: {data[CB_PIN]} Value: {data[CB_VALUE]} Time Stamp: {date}')
    compute_angle(data[CB_PIN],data[CB_VALUE],'ShAA')

# def Elbow_read():
board = telemetrix.Telemetrix(arduino_wait=2)


# setting Encoder input PINs
board.set_pin_mode_digital_input(DATA_PIN_Elbow, the_callback_elbow)
board.set_pin_mode_digital_input(STATE_PIN_Elbow, the_callback_elbow)

board.set_pin_mode_digital_input(DATA_PIN_ShFE, the_callback_shfe)
board.set_pin_mode_digital_input(STATE_PIN_ShFE, the_callback_shfe)

board.set_pin_mode_digital_input(DATA_PIN_ShAA, the_callback_shaa)
board.set_pin_mode_digital_input(STATE_PIN_ShAA, the_callback_shaa)



####################################################################################################################################################




def torque_control(joint,input):
    if joint == 'Elbow':
        # Setting Torque output PINs
        board.set_pin_mode_digital_output(DRIVER_PIN1_ELBOW)
        board.set_pin_mode_digital_output(DRIVER_PIN2_ELBOW)
        # Setting PIN values
        board.digital_write(DRIVER_PIN1_ELBOW, 0)
        board.digital_write(DRIVER_PIN2_ELBOW, 1)
        board.set_pin_mode_analog_output(PRESSURE_PIN_ELBOW)
        board.analog_write(PRESSURE_PIN_ELBOW, input)
    if joint == 'ShFE':
        board.set_pin_mode_digital_output(DRIVER_PIN1_SHFE)
        board.set_pin_mode_digital_output(DRIVER_PIN2_SHFE)
        board.digital_write(DRIVER_PIN1_SHFE, 0)
        board.digital_write(DRIVER_PIN2_SHFE, 1)
        board.set_pin_mode_analog_output(PRESSURE_PIN_SHFE)
        board.analog_write(PRESSURE_PIN_SHFE, input)        
    if joint == 'ShAA':
        board.set_pin_mode_digital_output(DRIVER_PIN1_SHAA)
        board.set_pin_mode_digital_output(DRIVER_PIN2_SHAA)
        board.digital_write(DRIVER_PIN1_SHAA, 0)
        board.digital_write(DRIVER_PIN2_SHAA, 1)
        board.set_pin_mode_analog_output(PRESSURE_PIN_SHAA)
        board.analog_write(PRESSURE_PIN_SHAA, input)
        
        
print('Enter Control-C to quit.')

# back to rest position
try:
    while True:
        time.sleep(.001)
        torque_control('ShFE',0)
        torque_control('ShAA',0)
        torque_control('Elbow',0)
except KeyboardInterrupt:
    board.shutdown()
    sys.exit(0)



##########################################     MAIN     #########################################################

# The max output torque for elbow is 200 for shfe is 256 and for shaa is 230
# # All joints moving together
# try:
#     while True:
#         print('Going up...')
#         for i in range(256):
#             torque_control('Elbow',i)
#             torque_control('ShFE',i)
#             torque_control('ShAA',i)
#             time.sleep(1)  # controlling the frequency
#             print(i)
#             print(int(count['Elbow']*360/1024))
#             if int(count['Elbow']*360/1024) == 30:
#                 for j in range(i, -1, -1):
#                     torque_control('Elbow',i)
#                     time.sleep(1)  # controlling the frequency                
#             if int(count['ShFE']*360/1024) == 20:
#                 for j in range(i, -1, -1):
#                     torque_control('ShFE',i)
#                     time.sleep(1)  # controlling the frequency 
#             if int(count['ShAA']*360/1024) == 20:
#                 for j in range(i, -1, -1):
#                     torque_control('ShAA',i)
#                     time.sleep(1)  # controlling the frequency  
# except KeyboardInterrupt:
#     board.shutdown()
#     sys.exit(0)

# All joints moving together
# try:
#     print('Going up...')
#     for i in range(200):
#         torque_control('ShFE',i)
#         time.sleep(2)  # controlling the frequency
#         torque_control('ShAA',i)
#         torque_control('Elbow',i)
#         print(i)
#     for i in range(200,-1,-1):
#         torque_control('ShFE',i)
#         time.sleep(2)  # controlling the frequency
#         torque_control('ShAA',i)
#         torque_control('Elbow',i)
#         print(i)
# except KeyboardInterrupt:
#     board.shutdown()
#     sys.exit(0)