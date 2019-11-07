import os
import matplotlib.pyplot as plt
'''
if os.path.isfile(os.path.join(os.path.join('HW1_dataset','groundtruth'),'gt{:0>6d}.png'.format(count))):
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(plt.imread(os.path.join(os.path.join('HW1_dataset','groundtruth'),'gt{:0>6d}.png'.format(count))))
    plt.subplot(1,2,2)
    fgMask=cv2.cvtColor(fgMask,cv2.COLOR_BGR2RGB)
    plt.imshow(fgMask,cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join('vs','{}.png'.format(count)))
    plt.close()
'''