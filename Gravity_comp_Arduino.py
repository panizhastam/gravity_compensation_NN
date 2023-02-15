import sys
import time
import torch
from telemetrix import telemetrix
from torch import nn
from math import pi


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
count = {'Elbow': 0, 
            'ShFE': 0,
            'ShAA': 0}
joint_dir = {'Elbow': 0, 
            'ShFE': 0,
            'ShAA': 0}



def compute_angle(pin_number,pin_value,joint):
    if joint == 'Elbow':        
        if pin_number == DATA_PIN_Elbow:
            if prev_pin[joint][0] != prev_pin[joint][1]:
                if prev_pin[joint][1] != pin_value:
                    count[joint] += 1
                    joint_dir[joint] = 0
                else:
                    count[joint] -= 1
                    joint_dir[joint] = 1
    if joint == 'ShFE':        
        if pin_number == DATA_PIN_ShFE:
            if prev_pin[joint][0] != prev_pin[joint][1]:
                if prev_pin[joint][1] == pin_value:
                    count[joint] += 1
                    joint_dir[joint] = 0
                else:
                    count[joint] -= 1
                    joint_dir[joint] = 1
    if joint == 'ShAA':        
        if pin_number == DATA_PIN_ShAA:
            if prev_pin[joint][0] != prev_pin[joint][1]:
                if prev_pin[joint][1] == pin_value:
                    count[joint] += 1
                    joint_dir[joint] = 0
                else:
                    count[joint] -= 1
                    joint_dir[joint] = 1
    s =""
    for x in count:
        s+= str(x) + ': '+ str(count[x]*360/1024) + "   " + 'dir: ' + str(joint_dir[x]) + "   "
        
    s+="\n"
    # print(s)
    prev_pin[joint][0] = pin_value
    
    #return [count['Elbow'], count['ShFE'], count['ShAA']]*360/1024

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

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = self.layer4(x)
        return x
    
def predict(arr):
    model = Network(6, 3)
    path = "trainedmodels/MLP1.pth"
    model.load_state_dict(torch.load(path))
    model.eval()    
    input = torch.tensor(arr)
    predicted_output = model(input)
    predicted_output.cpu().detach().numpy()
    el = int(predicted_output[0][0])
    fe = int(predicted_output[0][1]+30)
    aa = int(predicted_output[0][2]+40)
    print(str(el)+"   "+str(fe)+"   "+str(aa))
    torque_control(el,fe,aa)

    
    
def gravity_compensate():
    arr = [[count['Elbow']*2*pi/1024, count['ShFE']*2*pi/1024, count['ShAA']*2*pi/1024, joint_dir['Elbow'], joint_dir['ShFE'], joint_dir['ShAA']]]
    arr = torch.tensor(arr, dtype=torch.float32)
    model = Network(6, 3)
    path = "trainedmodels/MLP1.pth"
    model.load_state_dict(torch.load(path))
    model.eval()    
    input = torch.tensor(arr)
    predicted_output = model(input)
    predicted_output.cpu().detach().numpy()
    # el = int(predicted_output[0][0])
    # fe = int(predicted_output[0][1]+50)
    # aa = int(predicted_output[0][2]+70)
    el = int(count['Elbow']*2.8**360/1024)
    fe = int(count['ShFE']*2.8**360/1024)
    aa = int(count['ShAA']*2.8**360/1024)
    print(str(el)+"   "+str(fe)+"   "+str(aa))
    torque_control(el,fe,aa)

    # predict(arr)



board = telemetrix.Telemetrix(arduino_wait=2)


# setting Encoder input PINs
board.set_pin_mode_digital_input(DATA_PIN_Elbow, the_callback_elbow)
board.set_pin_mode_digital_input(STATE_PIN_Elbow, the_callback_elbow)

board.set_pin_mode_digital_input(DATA_PIN_ShFE, the_callback_shfe)
board.set_pin_mode_digital_input(STATE_PIN_ShFE, the_callback_shfe)

board.set_pin_mode_digital_input(DATA_PIN_ShAA, the_callback_shaa)
board.set_pin_mode_digital_input(STATE_PIN_ShAA, the_callback_shaa)



####################################################################################################################################################




def torque_control(el,fe,aa):

        # Setting Torque output PINs
        board.set_pin_mode_digital_output(DRIVER_PIN1_ELBOW)
        board.set_pin_mode_digital_output(DRIVER_PIN2_ELBOW)
        # Setting PIN values
        board.digital_write(DRIVER_PIN1_ELBOW, 0)
        board.digital_write(DRIVER_PIN2_ELBOW, 1)
        board.set_pin_mode_analog_output(PRESSURE_PIN_ELBOW)
        board.analog_write(PRESSURE_PIN_ELBOW, el)
        
        board.set_pin_mode_digital_output(DRIVER_PIN1_SHFE)
        board.set_pin_mode_digital_output(DRIVER_PIN2_SHFE)
        board.digital_write(DRIVER_PIN1_SHFE, 0)
        board.digital_write(DRIVER_PIN2_SHFE, 1)
        board.set_pin_mode_analog_output(PRESSURE_PIN_SHFE)
        board.analog_write(PRESSURE_PIN_SHFE, fe)
        
        board.set_pin_mode_digital_output(DRIVER_PIN1_SHAA)
        board.set_pin_mode_digital_output(DRIVER_PIN2_SHAA)
        board.digital_write(DRIVER_PIN1_SHAA, 0)
        board.digital_write(DRIVER_PIN2_SHAA, 1)
        board.set_pin_mode_analog_output(PRESSURE_PIN_SHAA)
        board.analog_write(PRESSURE_PIN_SHAA, aa)
        
        # print("end")
        

##########################################     MAIN     #########################################################

print('Enter Control-C to quit.')

# back to rest position
try:
    while True:
        time.sleep(.001)
        # gravity_compensate()
        # torque_control(0,0,0)
        el = int(abs(count['Elbow']*2*360/1024))
        fe = int(abs(count['ShFE']*3*360/1024))
        aa = int(abs(count['ShAA']*3*360/1024))
        print(str(el)+"   "+str(fe)+"   "+str(aa))
        torque_control(el,fe,aa)
except KeyboardInterrupt:
    board.shutdown()
    sys.exit(0)


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
# try:
#     while True:
#         # x = out.values()
#         # x = list(x)
#         # x = np.array(x)
#         # x = x.reshape(1,3)
#         # teout = torch.Tensor(x)
#         # result = loaded_model.predict(teout)
#         # print(result)
#         board.analog_write(PRESSURE_PIN_ELBOW, 100)
#         time.sleep(.00001)
# except KeyboardInterrupt:
#     board.shutdown()
#     sys.exit(0)




