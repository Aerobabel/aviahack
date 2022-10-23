import torch
import cv2
model = torch.hub.load('yolov5', 'yolov5n', source= 'local')

cap = cv2.VideoCapture('cool.mp4')
#cap = cv2.VideoCapture(0)

# focal length finder function
def FocalLength(measured_distance, real_width, width_in_rf_image):
	focal_length = (width_in_rf_image* measured_distance)/ real_width
	return focal_length

# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
	distance = (real_face_width * Focal_Length)/face_width_in_frame
	return distance

while True:
    img = cap.read()[1]
    if img is None:
        break
    result = model(img)
    df = result.pandas().xyxy[0]

    for ind in df.index:
        x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
        x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
        label = df['name'][ind]

        conf = df['confidence'][ind]
        text = label + ' ' + str(conf.round(decimals=2))
        coll = 'possible collision'




        if label == "airplane":

            assess = FocalLength(200,1000,300)
            dis = Distance_finder(assess, 1000, y2-y1)
            label = label + " " + str(int(dis)) + "m"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            if dis <= 100:
                cv2.putText(img, coll, (x1+20, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)









        #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        #cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)


    cv2.imshow('video', img)
    cv2.waitKey(10)


'''
img = cv2.imread('test.jpg')
result = model(img)

df = result.pandas().xyxy[0]

print(df)

for ind in df.index:
    x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
    x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
    label = df['name'][ind]

    cv2.rectangle(img,(x1,y1), (x2,y2), (255,255,0),2)
    cv2.putText(img,label,(x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0),2)

cv2.imshow('IMAGE', img)
cv2.waitKey(0)

'''


