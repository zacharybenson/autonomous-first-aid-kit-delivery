 ---

<div align="center">    
 
# Autonomous First Aid Kit Delievery ⛑️
 
</div>

<div align="left">

## Overview  
  
Suppose a downed pilot is lost, stranded in an area where we only have an idea of their location. 
 This individual is injured, must receive medical supplies, and then be extracted ASAP.  
We need to devise a way, using a drone and AI, to locate this person, deliver much-needed first aid, 
 and then report their location to an extraction.  What’s involved with locating and identifying a stranded individual and then delivering a first aid package to that person? 
 There are several main points to consider:
 - While the drone is deployed and fling a predetermined flight path over a defined air space, 
 - how might we locate and identify a person from the air?  How may we track the objective after it is verified?  
    - Perhaps we could create a refined object tracker to track.  How might we estimate the stranded individual’s geolocation?  
 - Finally, how might we deliver a package once the person’s location has been determined?  


## Objective
  
The objective for this project is to lift a care package attached to a cable, send the drone off, using a predefined flight path & pattern, locate and verify the stranded person 
  (dressed, male mannequin) on the ground, safely deliver the first aid packaged within ten feet of that person and return home.  
  Your team will put together all the code necessary to carry out a successful mission.  
  You may use any code we’ve covered and/or created throughout this course to help.  
  You must determine how to prepare and implement your object recognition model, how you will locate the person in need of help, and how to deliver the care package undamaged. 
  You already have some of the code modules from the previous PEX that you will need for this project: drone_lib.py, RealSense camera routines, etc...  
  Start a new PyCharm project starting with those elements and create a new drone_mission.py to hold your main project code.  
  An initial yolo object recognition model will be provided for you (see section Problem#2: Object Recognition Model for details) and this document contains some helpful details as well. 
  The rest will be up to your team to figure out.  While some aspects of this project cannot be easily tested via SITL, much of the important pieces can.   
  It’s up to you to discover how SITL can help as you progress through this project.


## Main project elements

- Use object recognition instead for target identification.
- The use a sophisticated object tracker for positioning over objective.
- Calculate GIS location for package drop rather than relying on non-deterministic algorithm.
- Lowering a package to the ground and detaching from it.


