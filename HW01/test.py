import os

gt=os.listdir(os.path.join('HW1_dataset','groundtruth'))
gt=[int(path[5:8]) for path in gt]
print(gt)