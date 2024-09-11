# Fraud Detection in video when the invoice is not returned to the customer after receiving the cash
All the codes used for getting to the final approach were done on colab
This repository contains three colab notebooks - 
1. Retail_Assignment_Pre_processing.ipynb - this contains the codes that demonstrate all the different techniques I tried out 
2. Retail_Assignment_yolo.ipynb - this contains the codes to train the yolov8 model to recognise the note
3. Retail_Assignment_Implementation.ipynb - this contains the code of the final implementation

Additionally, I have also uploaded main.py which can be used to run the inference on a video and the code saves the video with the inference 

All the output videos demonstrating the fraud detection algorithm can be found in this folder - https://drive.google.com/drive/folders/1MsNXsCmmcv5OCDtqwzOT0g3EtzUWC0Le?usp=sharing
1. cash_no_invoice_30sec_final.mp4 - This video demonstrates fraud detection with counter
2. multiple_cash_transaction_raw_example_final.mp4 - This video demonstrates non-fraud detection transactions with counter
3. cash_no_invoice_30sec_detection.mp4 and multiple_cash_transaction_raw_example_detection.mp4 - These videos can be used to better understand how the code is working 
