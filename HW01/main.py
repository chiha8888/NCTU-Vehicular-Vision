import cv2
import os
import matplotlib.pyplot as plt
from postprocess import postprocess
from preprocess import preprocess


# Path Setting
print("Use Default Path Setting Press 0")
print("Custom Path Setting Press 1")
setting = int(input(">>>"))
if setting == 1:
    ground_truth_path = str(input("Your Ground Truth Data Path:"))
    input_data_path = str(input("Your Input Data Path:"))
    output_data_path = str(input("Your Output Save Path:"))
    ROI_path = str(input("Your ROI Path & Name:"))
elif setting == 0:
    ground_truth_path = "./HW1_dataset/groundtruth"
    input_data_path = "./HW1_dataset/input"
    output_data_path = "./EvaluationV2/results"
    ROI_path = "./HW1_dataset/ROI.bmp"
else:
    os._exit(0)

# Read Files
gt=os.listdir(os.path.join(ground_truth_path))
gt=[int(path[5:8]) for path in gt]

mask = cv2.imread(os.path.join(ROI_path), 0)

backSub = cv2.bgsegm.createBackgroundSubtractorMOG(history=260, backgroundRatio=0.4)
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(os.path.join(os.path.join(input_data_path), 'in%06d.png')))
if capture.isOpened() == 0:
    print('Unable to Open File')
    exit(0)


count = 1
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(preprocess(frame, mask))
    fgMask = postprocess(fgMask, 25, 20)

    cv2.imshow('frame', frame)
    cv2.imshow('fgMask', fgMask)
    plt.show()
    cv2.waitKey(10)
    #save result (comparing with ground_truth)
    if count in gt:
        cv2.imwrite(os.path.join(output_data_path, '{:0>6d}.png'.format(count)), fgMask)

    count += 1

print ("Finish! Press Any Keys to Exit")    
cv2.waitKey(0)
cv2.destroyAllWindows()