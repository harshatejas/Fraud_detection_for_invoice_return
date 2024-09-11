# All the commented codes are used for visualization
import cv2
import os
import numpy as np
from ultralytics import YOLO

input_video_path = "cash_no_invoice_30sec.mp4"
output_video_path = "cash_no_invoice_30sec_final.mp4"

model_pose = YOLO("yolov8x-pose.pt")
model_note_recog = YOLO("weights/best.pt")

cap = cv2.VideoCapture(input_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(input_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

cash_count = 0
invoice_count = 0
fraud_flag = False
money_received = False

roi = (300, 350, 800, 1000)
point_on_right_vertical_line = (800, 675) # To determine the biller
point_on_left_vertical_line_1 = (300, 512) # To determine the customer
point_on_left_vertical_line_2 = (300, 837) # To determine the customer

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, img = cap.read()

    if not ret:
        break

    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    timestamp_seconds = frame_index / fps

    # Draw ROI
    #cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 255) , 2)

    results = model_pose.predict(img, verbose = False)
    class_names = model.names

    biller_wrist = {}
    customer_wrist = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        keypoints_data = result.keypoints.data.cpu().numpy()

    for box, cf, cls, keypoint_data in zip(boxes, conf, classes, keypoints_data):
        if cls == 0:
            box = [int(x) for x in box]
            x1, y1, x2, y2 = box

            #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255) , 2)
            class_name = class_names[cls]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text = str(f'{class_name} {cf:.2f}')

            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size

            #cv2.rectangle(img, (box[0], box[1] - text_height - thickness - 5), (box[0] + text_width, box[1]), (0, 0, 255), -1)

            #cv2.putText(img, str(f'{class_name} {cf:.2f}'), (int(box[0]), int(box[1])-5), font, font_scale, (255, 255, 255), thickness)

            if keypoint_data is not None:
                '''
                for i, (x, y, confidence) in enumerate(keypoint_data):
                    if confidence > 0.5:
                        if i == 9 or i == 10: # Left and Right Wrist
                            if roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]: # Inside ROI
                                #cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                            else: # Outside ROI
                                #cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

                        else:
                            #cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                '''
                if is_point_inside_roi(point_on_right_vertical_line, box):
                    for i, (x, y, confidence) in enumerate(keypoint_data):
                        if confidence > 0.5:
                            if i == 9: # Left  Wrist
                                #cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
                                biller_wrist['left'] = (int(x), int(y))
                            elif i == 10: # Right Wrist
                                #cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
                                biller_wrist['right'] = (int(x), int(y))

                if is_point_inside_roi(point_on_left_vertical_line_1, box) or is_point_inside_roi(point_on_left_vertical_line_2, box):
                    for i, (x, y, confidence) in enumerate(keypoint_data):
                        if confidence > 0.5:
                            if i == 9: # Left  Wrist
                                #cv2.circle(img, (int(x), int(y)), 5, (0, 0, 0), -1)
                                customer_wrist.append(['left', int(x), int(y)])
                            elif i == 10: # Right Wrist
                                #cv2.circle(img, (int(x), int(y)), 5, (0, 0, 0), -1)
                                customer_wrist.append(['right', int(x), int(y)])



    result_notes = model_note_recog(img, save = False, conf = 0.5, verbose = False)
    for result in result_notes:
        boxes = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

    for box, cf, cls in zip(boxes, conf, classes):
        box = [int(x) for x in box]
        x1, y1, x2, y2 = box

        #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0) , 2)

        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)

        if not money_received:
            for key, value in biller_wrist.items():
                distance = math.sqrt((value[0] - midpoint[0])**2 + (value[1] - midpoint[1])**2)
                #cv2.line(img, (int(midpoint[0]), int(midpoint[1])), (int(value[0]), int(value[1])), (255, 255, 255), 2)

                mid_x = (int(midpoint[0]) + int(value[0])) // 2
                mid_y = (int(midpoint[1]) + int(value[1])) // 2

                #cv2.putText(img, f'{distance:.2f}', (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                if distance < 110:
                    #print("Money Received!!")
                    '''
                    text = "Money Received!!"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    font_color = (0, 255, 0)
                    thickness = 3
                    line_type = cv2.LINE_AA

                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    position = (img.shape[1] - text_width - 10, text_height + 10)

                    rectangle_bgr = (0, 0, 0)
                    cv2.rectangle(img, (position[0] - 5, position[1] - text_height - 5),
                                    (position[0] + text_width + 5, position[1] + 5),
                                    rectangle_bgr, cv2.FILLED)

                    cv2.putText(img, text, position, font, font_scale, font_color, thickness, line_type)
                    '''
                    cash_count += 1
                    money_received = True
                    start_time = timestamp_seconds

        if money_received and (timestamp_seconds - start_time) > 1:
            for side, x, y in customer_wrist:
                distance = math.sqrt((x - midpoint[0])**2 + (y - midpoint[1])**2)
                #cv2.line(img, (int(midpoint[0]), int(midpoint[1])), (int(x), int(y)), (255, 255, 255), 2)

                mid_x = (int(midpoint[0]) + int(x)) // 2
                mid_y = (int(midpoint[1]) + int(y)) // 2

                #cv2.putText(img, f'{distance:.2f}', (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                if distance < 110:
                    #print("Transaction Done!!!!")
                    '''
                    text = "Transaction Done!!!!"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    font_color = (0, 255, 0)
                    thickness = 3
                    line_type = cv2.LINE_AA

                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    position = (img.shape[1] - text_width - 10, text_height + 10)

                    rectangle_bgr = (0, 0, 0)
                    cv2.rectangle(img, (position[0] - 5, position[1] - text_height - 5),
                                    (position[0] + text_width + 5, position[1] + 5),
                                    rectangle_bgr, cv2.FILLED)

                    cv2.putText(img, text, position, font, font_scale, font_color, thickness, line_type)
                    '''
                    invoice_count += 1
                    money_received = False

    if money_received:
        if timestamp_seconds - start_time > 15:
            text = "Fraud Detected"
            '''
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_color = (0, 255, 0)
            thickness = 3
            line_type = cv2.LINE_AA

            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            position = (img.shape[1] - text_width - 10, text_height + 10)

            rectangle_bgr = (0, 0, 0)
            #cv2.rectangle(img, (position[0] - 5, position[1] - text_height - 5),
            #                (position[0] + text_width + 5, position[1] + 5),
            #                rectangle_bgr, cv2.FILLED)

            #cv2.putText(img, text, position, font, font_scale, font_color, thickness, line_type)
            '''
            fraud_flag = True

    # Display the text on the image
    if fraud_flag:
        fraud_alert = "Yes"
    else:
        fraud_alert = "No"

    text_lines = [f'Cash: {cash_count} ', f'Invoice: {invoice_count}', f'Fraud: {fraud_alert}']
    height, width, _ = img.shape

    x1 = width - 200
    y1 = 10
    x2 = width
    y2 = 150

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    line_height = 40
    for i, line in enumerate(text_lines):
        text_x = x1 + 10
        text_y = y1 + (i + 1) * line_height
        cv2.putText(img, line, (text_x, text_y), font, font_scale, (0, 255, 0), 2)

    out.write(img)

cap.release()
out.release()