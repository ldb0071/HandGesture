<!-- Create a readme for home automation using hand gesture recognation -->
# Home automation using Hand Gesture Recognition 

Mohamed Khider University of Biskra 
Faculty of Sciences and Technology 
Electrical Engineering Department  

Master 2 in Electrical Engineering  
Supervisor from Algeria: Ouafi abdelkarim
Supervisor from France: Taleb Abdelmalik 
Student: Sellam Abdellah Zakaria
___

## Introduction
A home automation system simplifies the operation of various household appliances while 
still conserving electricity. Home automation or building automation, which is based on the 
energy-saving principle, has made life much easier in recent years. It entails the automatic control 
of all electrical or electronic devices in the household, as well as remote control through wireless 
communication. The home automation system allows for centralized control of lighting, air 
conditioning and heating, audio/video systems, surveillance systems, kitchen appliances, and all 
other equipment used in home systems.
___
## Projects Steps 
The project is based on a deep learning model for hand gesture recognition and a mediapipe library with the addition of frame background subtraction technology to detect the hand and the gesture. The project is based on the following steps:

### Deep learning model
1.	Collecting the data set
2.	Training the model
3.	Testing the model
### System
1.  starting the system
2.	Background subtraction
3.	Hand detection
4.	Hand tracking
5.	Hand gesture recognition
sending the command to the raspberry pi
6.	Action execution

___
## Background subtraction using frame differance method
Action recognition is just one example of the many computer vision systems that rely on foreground/background separation. Sudden shifts in lighting, shadows, camera shakes, and the presence of moving or changing objects in the background (such as trees or screens) present significant difficulties for background subtraction techniques. Temporal diffencing takes advantage of two or more successive frames to identify and isolate areas of motion. If the temporal changes are caused by noise or illumination change due to weather conditions, this method is susceptible to false detection.
___
## Hardware Requirements
1. Raspberry Pi 3
2. Camera Module
3. Relay Module
4. Jumper Wires
5. Breadboard
6. led
7. servo motor
___
## Os
2. Windows
3. Raspbian OS
___
## requirements 
```bash 
pip install -r requirements.txt
```
___

## Classes
```bash
CLASS_MAP = {0:'fist',
   1:'five',
   2:'none',
   3:'okay',
   4:'L',
   5:'rad',
   6:'three',
 7:'thumbs'}
 ```
