import numpy as np

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.classes = classes

    def view(self):
        print('xmin:{} ymin:{}'.format(self.xmin, self.ymin))
        print('xmax:{} ymax:{}'.format(self.xmax, self.ymax))
        print('classes:{}'.format(self.classes))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_netout(netout, anchors, threshold, net_h, net_w):
    '''
    netout: (grid_h, grid_w, #anchors*(x+y+w+h+p+#classes))
    achors: 3
    net_h=net_w=416
    '''
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))  # (grid_h,grid_w,3,85)
    nb_class = netout.shape[-1] - 5  # nb_class=80

    netout[..., :2] = sigmoid(netout[..., :2])  # x,y
    netout[..., 4:] = sigmoid(netout[..., 4:])  # p,C1,C2...,C80
    netout[..., 5:] = np.expand_dims(netout[..., 4], axis=-1) * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > threshold

    boxes = []
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):

                p = netout[row][col][b][4]
                if p < threshold:
                    continue
                x, y, w, h = netout[row][col][b][:4]
                x = (col + x) / grid_w  # center position, unit: image width
                y = (row + y) / grid_h  # center position, unit: image height
                w = anchors[2 * b] * np.exp(w) / net_w  # unit: image width
                h = anchors[1 + 2 * b] * np.exp(h) / net_h  # unit: image height
                classes = netout[int(row)][col][b][5:]

                boxes.append(BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, classes))

    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def iou(box_i, box_j):
    x1, y1 = max(box_i.xmin, box_j.xmin), max(box_i.ymin, box_j.ymin)
    x2, y2 = min(box_i.xmax, box_j.xmax), min(box_i.ymax, box_j.ymax)
    if x1 > x2 or y1 > y2:
        return 0
    intersect = (x2 - x1) * (y2 - y1)
    area_i = (box_i.xmax - box_i.xmin) * (box_i.ymax - box_i.ymin)
    area_j = (box_j.xmax - box_j.xmin) * (box_j.ymax - box_j.ymin)
    union = area_i + area_j - intersect
    return intersect / union

def do_nms(boxes, threshold):
    if len(boxes) == 0:
        return
    dic = {'person': 0, 'car': 2}  # by COCO dataset
    for name, c in dic.items():

        index_sort = np.argsort([-box.classes[c] for box in boxes])  # big->small
        for i in range(len(index_sort)):
            box_i = boxes[index_sort[i]]
            if box_i.classes[c] == 0:
                continue
            for j in range(i + 1, len(index_sort)):
                box_j = boxes[index_sort[j]]
                if iou(box_i, box_j) > threshold:
                    box_j.classes[c] = 0

def get_boxes(boxes, threshold):
    dic = {'person': 0, 'car': 2}  # by COCO dataset
    bbs = []  # [(lu,rd,name,score),(lu,rd,name,score),...]
    for box in boxes:
        for name, i in dic.items():
            if box.classes[i] > threshold:
                bbs.append(((box.xmin, box.ymin), (box.xmax, box.ymax), name, box.classes[i] * 100))
                # don't break, many labels may trigger for one box
    return bbs