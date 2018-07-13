##//Automotive detection of situation criteria for traffic classification##
import numpy as np
import cv2
import math
import threading
threshold = 200
def distance(x1,y1):
    if distance [i][j] >= 0.5:
        mid_x1 = len(detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))) /2
        mid_y1 = len(detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))) /2
        apx_dist = round((1-(detection_box [0][i][2]-detection_box [0][i][1]))**4,1)
        cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x1*800),int(mid_y1*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if apx_distance <=0.5:
            if mid_x > 0.3 and mid_x < 0.7:
                cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u
'''detection precosse of outgoing vehicle and stause of road whether ts safe or not , assiganing the rules for best solution for developing it helpfull
'''
#========================== DetectSituativ_OutGoingVehicles_OGV_Method ===================================================================================#
def detectSituativ_OutGoingVehicles_OGV( image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = 200
    cascade = cv2.CascadeClassifier('Data/vback.xml')
    #cascade = cv2.CascadeClassifier('Data/cars2.xml')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #Create CLAHE object
    clahe_image = clahe.apply(gray_image) #Apply CLAHE to grayscale image from webcam
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.9, minNeighbors=26, minSize = (50,50),maxSize = (300,290) )  
  
    if len(features) == 0:
            return []
    else: 
                features[:, 2:] += features[:, :2] 
                detection_box = [] 
                for i in features: 
                    detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))
                for j in detection_box:    
                        x_coordinate, y_coordinate, width_position, height_position = j 
                        cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate+width_position, y_coordinate+height_position), (255, 255, 255), 2)
                        if len (detection_box) == 1:
                            cv2.putText(frame, "SAfe-OGV", (x_coordinate, y_coordinate+height_position+13),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0),1)
                        elif len (detection_box) == 2:
                                cv2.putText(frame, "Regular-OGV", (x_coordinate, y_coordinate+height_position),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0),1)
                        elif len (detection_box) >= 3:
                                cv2.putText(frame, "Ctitical-OGV", (x_coordinate, y_coordinate+height_position),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0),1)

                        v = y_coordinate + height_position - 5
                        roi = gray_image[y_coordinate+10:y_coordinate + height_position-10, x_coordinate+10:x_coordinate + width_position-10]
                        #mask = cv2.GaussianBlur(roi, (25, 25), 2)
                        mask = cv2.bilateralFilter(roi,5,65,65)
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                
                        # check if light is onif maxVal - minVal > threshold:
                        if maxVal - minVal > threshold:
                    
                                cv2.circle(roi, maxLoc, 10, (255, 0, 0), 2)
                    
                                    # Red light
                                if 1.0/8*(height_position-30) < maxLoc[1] < 4.0/8*(height_position-30):
                                   x_coordinate, y_coordinate, width_position, height_position = j
                                   cv2.putText(image, 'brake light On', (x_coordinate+5, y_coordinate-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        #self.red_light = True
                #if len (features) >= 4:
                    #cv2.putText(image, 'Danger-OGV', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                #elif len (features) == 3:
                    #cv2.putText(image, 'Criical-OGV', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #else:
                    #cv2.putText(image, 'Vehicle is not at  highway', (x_val, y_val-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #cv2.waitKey(0)&0xFF

                #elif len(features) == 2:
                     #cv2.putText(image, 'Regular-OGV', (x_coordinate, y_coordinate-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     
               # elif len(features)<= 1:
                   #  cv2.putText(image, 'Safe-OGV', (x_coordinate, y_coordinate-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                         
                     #cv2.waitKey(0)&0xFF'''

                

                
#========================== detectSituativ_IncomingVehicles_ICV =====================================================================================#
def detectSituativ_IncomingVehicles_ICV( image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    cascade = cv2.CascadeClassifier('Data/fv1.xml')
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #Create CLAHE object
    #clahe_image = clahe.apply(gray_image) #Apply CLAHE to grayscale image from webcam
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=15, minSize = (40,40),maxSize = (280,150) )  
  
    
    if len(features) == 0: 
        return []
    else: 
        features[:, 2:] += features[:, :2] 
        
        detection_box = [] 
        for i in features: 
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))   
        
        for j in detection_box:
                x_coordinate, y_coordinate, width_position, height_position = j
                cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width_position, y_coordinate + height_position), (0,255,0), 2)
                roi = gray_image[y_coordinate:y_coordinate + height_position, x_coordinate:x_coordinate + width_position]
                mask = cv2.GaussianBlur(roi, (15, 15), 1)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                #cv2.putText(frame, "IncominVehicle", (x_coordinate, y_coordinate+height_position+13),cv2.FONT_HERSHEY_TRIPLEX, .5, (0,255,0))
                if len (features) >= 4:
                    cv2.putText(image, 'incomingDanger-V', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif len (features) == 3:
                    cv2.putText(image, 'incomingCritical-V', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #else:
                    #cv2.putText(image, 'Vehicle is not at  highway', (x_val, y_val-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #cv2.waitKey(0)&0xFF

                elif len(features) == 2:
                     cv2.putText(image, 'IncomingRegular-V', (x_coordinate, y_coordinate-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     
                elif len(features)<= 1:
                     cv2.putText(image, 'incomingSafe-V', (x_coordinate, y_coordinate-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                         
                    # cv2.waitKey(0)&0xFF


#========================== detectSituativ_pedestrians==============================================================================================#
def detectSituativ_pedestrians( image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    cascade = cv2.CascadeClassifier('Data/MyPedestrian.xml') 
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=6,minSize = (30,90),maxSize = (90,180)) 
    
    if len(features) == 0: 
        return []
    else: 
        features[:, 2:] += features[:, :2] 
        detection_box = [] 
        for i in features: 
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))
        for j in detection_box:
                x_coordinate, y_coordinate, width_position, height_position = j 
                cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width_position, y_coordinate + height_position), (0,0,0), 2)
                roi = gray_image[y_coordinate:y_coordinate + height_position, x_coordinate:x_coordinate + width_position]
                mask = cv2.GaussianBlur(roi, (15, 15), 1)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                if len (detection_box) >= 4:
                    cv2.putText(image, 'Danger-P', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif len (features) == 3:
                    cv2.putText(image, 'Criical-P', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #else:
                    #cv2.putText(image, 'Vehicle is not at  highway', (x_val, y_val-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #cv2.waitKey(0)&0xFF

                elif len(features) == 2:
                     cv2.putText(image, 'Regular-P', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     
                elif len(features)<= 1:
                     cv2.putText(image, 'Safe-P', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                #cv2.putText(image, "Pedestrian", (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                


#========================== detectSituativ_VechicleCrossing_VC ==========================================================================================#
def detectSituativ_VechicleCrossing_VC( image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    cascade = cv2.CascadeClassifier('Data/carside2.xml') 
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.4, minNeighbors=15,minSize = (120,60),maxSize = (300,250)) 
    
    if len(features) == 0: 
        return []
    else: 
        features[:, 2:] += features[:, :2] 
        detection_box = [] 
        for i in features: 
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))
        for j in detection_box:
                x_coordinate, y_coordinate, width_position, height_position = j 
                cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width_position, y_coordinate + height_position), (0,0,0), 2)
                roi = gray_image[y_coordinate:y_coordinate + height_position, x_coordinate:x_coordinate + width_position]
                mask = cv2.GaussianBlur(roi, (15, 15), 1)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                if len (features) >= 4:
                    cv2.putText(image, 'Vehicle crossing', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif len (features) == 3:
                    cv2.putText(image, 'Criical-VC', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #else:
                    #cv2.putText(image, 'Vehicle is not at  highway', (x_val, y_val-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #cv2.waitKey(0)&0xFF

                elif len(features) == 2:
                     cv2.putText(image, 'Regular-VC', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     
                elif len(features)<= 1:
                     cv2.putText(image, 'Safe-VC', (x_coordinate, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#================ ==============  detectSituativ_WarningSign =================================================================================#                  
def detectSituativ_WarningSign( image ):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    cascade = cv2.CascadeClassifier('Data/warning.xml')
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5,minSize = (35,35),maxSize = (200,200) )
    if len(features) == 0: 
        return []
    else: 
        features[:, 2:] += features[:, :2] 
        detection_box = [] 
        for i in features: 
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))        
        
        for j in detection_box:
                x_coordinate, y_coordinate, width_position, height_position = j 
                cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width_position, y_coordinate + height_position), (0,255,0), 2)
                roi = gray_image[y_coordinate:y_coordinate + height_position, x_coordinate:x_coordinate + width_position]
                mask = cv2.GaussianBlur(roi, (15, 15), 1)
                #cv2.putText(frame, , (x_coordinate, y_coordinate+height_position+13),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0))

#================ ==============  detectSituativ_GiveawaySign =================================================================================#                  
def detectSituativ_giveawaySign( image ):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    cascade = cv2.CascadeClassifier('Data/gvway.xml')
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3,minSize = (30,30),maxSize = (100,100) )
    if len(features) == 0: 
        return []
    else: 
        features[:, 2:] += features[:, :2] 
        detection_box = [] 
        for i in features: 
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))        
        
        for j in detection_box:
                x_coordinate, y_coordinate, width_position, height_position = j 
                cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width_position, y_coordinate + height_position), (0,255,0), 2)
                roi = gray_image[y_coordinate:y_coordinate + height_position, x_coordinate:x_coordinate + width_position]
                mask = cv2.GaussianBlur(roi, (15, 15), 1)
                #cv2.putText(frame, "Give a way ", (x_coordinate, y_coordinate+height_position+13),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0))
                
#========================== ===================detectSituativ_PrioritySign ============================================================================#
def detectSituativ_PrioritySign( image ):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    cascade = cv2.CascadeClassifier('Data/priorityroad.xml')
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=8, minSize = (40,40),maxSize = (200,200) )
        
    if len(features) == 0: 
        return []
    else: 
        features[:, 2:] += features[:, :2] 
        detection_box = [] 
        for i in features: 
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))       
       
        for j in detection_box:
                x_coordinate, y_coordinate, width_position, height_position = j 
                cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width_position, y_coordinate + height_position), (255,0,0), 2)
                roi = gray_image[y_coordinate:y_coordinate + height_position, x_coordinate:x_coordinate + width_position]
                mask = cv2.GaussianBlur(roi, (15, 15), 1)
                cv2.putText(frame, "priority", (x_coordinate, y_coordinate+height_position+13),cv2.FONT_HERSHEY_TRIPLEX, .5, (255,0,0))
#========================== detectSituativ_StopSign =========================================================================================#
def detectSituativ_StopSign( image ):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
   
    cascade = cv2.CascadeClassifier('Data/stop_sign.xml')
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=8, minSize = (35,35),maxSize = (200,200))
        
    if len(features) == 0: 
        return []
    else: 
        features[:, 2:] += features[:, :2] 
        detection_box = [] 
        for i in features: 
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))        
        
        for j in detection_box:
                x_coordinate, y_coordinate, width_position, height_position = j 
                cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width_position, y_coordinate + height_position), (128,0,128), 2)
                roi = gray_image[y_coordinate:y_coordinate + height_position, x_coordinate:x_coordinate + width_position]
                mask = cv2.GaussianBlur(roi, (15, 15), 1)
                cv2.putText(frame, "stop", (x_coordinate, y_coordinate+height_position+13),cv2.FONT_HERSHEY_TRIPLEX, .5, (128,0,128))
                
#========================== detectSituativ_DirectionalSign =============================================================================#

def detectSituativ_DirectionalSign( image ):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
   
    cascade = cv2.CascadeClassifier('Data/directionblue.xml')
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=4, minSize = (30,30),maxSize = (200,200))
        
    if len(features) == 0: 
        return []
    else: 
        features[:, 2:] += features[:, :2] 
        detection_box = [] 
        for i in features: 
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))        
        
        for j in detection_box:
                x_coordinate, y_coordinate, width_position, height_position = j 
                cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width_position, y_coordinate + height_position), (184,134,11), 2)
                cv2.putText(frame, "Directional", (x_coordinate, y_coordinate+height_position+13),cv2.FONT_HERSHEY_TRIPLEX, .5, (184,134,11))
                roi = gray_image[y_coordinate:y_coordinate + height_position, x_coordinate:x_coordinate + width_position]
                mask = cv2.GaussianBlur(roi, (15, 15), 1)
                
#============================= detectSituativ_TrafficSignals =============================================================================#
                
def detectSituativ_TrafficSignals( image ):

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = 200
        cascade = cv2.CascadeClassifier('Data/traffic_light.xml')
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #Create CLAHE object
        #clahe_image = clahe.apply(gray_image) #Apply CLAHE to grayscale image from webcam

        
        features = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3,minSize=(5, 10),maxSize=(120,120)) 
    
        if len(features) == 0:
                return []
        else: 
                features[:, 2:] += features[:, :2] 
                detection_box = [] 
                for i in features: 
                    detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))
                for j in detection_box:
                        x_coordinate, y_coordinate, width_position, height_position = j 
                        cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate+width_position, y_coordinate+height_position), (255, 255, 255), 2)
                        if len (detection_box) == 1:
                            cv2.putText(frame, "TrafficSignal", (x_coordinate, y_coordinate+height_position+13),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0),1)
                        elif len (detection_box) == 2:
                                cv2.putText(frame, "TrafficSignal-Intersection", (x_coordinate, y_coordinate+height_position),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0),1)
                        elif len (detection_box) >= 3:
                                cv2.putText(frame, "TrafficSignal-Critical Intersection", (x_coordinate, y_coordinate+height_position),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0),1)

                        v = y_coordinate + height_position - 5
                        roi = gray_image[y_coordinate+10:y_coordinate + height_position-10, x_coordinate+10:x_coordinate + width_position-10]
                        mask = cv2.GaussianBlur(roi, (25, 25), 2)
                        #mask = cv2.bilateralFilter(roi,5,65,65)
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                
                        # check if light is onif maxVal - minVal > threshold:
                        if maxVal - minVal > threshold:
                    
                                cv2.circle(roi, maxLoc, 5, (255, 0, 0), 2)
                    
                                    # Red light
                                if 1.0/8*(height_position-30) < maxLoc[1] < 1.5/8*(height_position-30):
                                #if 1.0/8*(height_position-30) < maxLoc[1] < 3.0/8*(height_position-30):
                                   x_coordinate, y_coordinate, width_position, height_position = j
                                   cv2.putText(image, 'RedLight-Stop', (x_coordinate+5, y_coordinate-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        #self.red_light = True
                    
                                            # Green light
                                #elif 3.5/8*(height_position-30) < maxLoc[1] <( height_position-30):
                                elif 2.5/8*(height_position-30) < maxLoc[1] <( height_position-30):
                                     x_coordinate, y_coordinate, width_position, height_position = j
                                     
                                     cv2.putText(image, 'GreenLight-Go-Normal_speed', (x_coordinate+5, y_coordinate - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        #self.green_light = True
    
                                                # yellow light
                                #elif 2.2/8*(height_position-30) < maxLoc[1] < 3.5/8*(height_position-30):
                                elif 1.5/8*(height_position-30) < maxLoc[1] < 2.5/8*(height_position-30):
                                     x_coordinate, y_coordinate, width_position, height_position = j
                                     cv2.putText(image, 'YellowLight-Slowdown', (x_coordinate+5, y_coordinate-10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                        #self.yellow_light = True'''
       
       

       
       

# capturing frame by frame
video = cv2.VideoCapture('b12.MP4')
#video = cv2.VideoCapture(2)




while(video.isOpened()):
    
    ret, frame = video.read() 
    #if ret==True:
        #frame = cv2.flip(frame,0)
       # out.write(frame)
    #frame = cv2.resize(frame, None,fx=0.4, fy=0.4, interpolation = cv2.INTER_LINEAR)                            

    if frame is None:
        cv2.destroyAllWindows() 
        break
    else: # at each frame read....
        detectSituativ_OutGoingVehicles_OGV(frame)  
        detectSituativ_IncomingVehicles_ICV (frame)
        detectSituativ_VechicleCrossing_VC (frame)
        detectSituativ_pedestrians (frame)
        detectSituativ_WarningSign (frame)
        detectSituativ_PrioritySign (frame)
        detectSituativ_StopSign (frame)
        detectSituativ_DirectionalSign (frame)
        detectSituativ_TrafficSignals (frame)
        detectSituativ_giveawaySign(frame)
        #if you want check the frame by frame.to see accurate detection un comment below comments
        
        #cv2.waitKey(0)
        #if 0xFF in (ord('p'),ord('l')):
         #   pass
        cv2.imshow('frame',frame)
        #cv2.imshow ('frame',gray_image)
    if cv2.waitKey(1) & 0xFF in (ord('q'),0x1B,0x0D): 
        break
video.release() # When everything done, release the capture...
#out.release()
cv2.destroyAllWindows() # closing the display window automatically...
