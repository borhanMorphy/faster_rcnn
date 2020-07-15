

def test_anchors():
    from utils.box import generate_anchors
    from cv2 import cv2
    import numpy as np

    blank_white = np.ones((600,600,3), dtype=np.uint8) * 255

    anchors = generate_anchors(16)

    print("anchors:")
    for anchor in anchors:
        print(anchor)
    for offset in range(300,330,30):
        boxes = anchors + offset
        for x1,y1,x2,y2 in boxes.long().numpy():
            blank_white = cv2.rectangle(blank_white, (x1,y1), (x2,y2), (0,0,255), 1)

    cv2.imshow("anchor tests", blank_white)
    cv2.waitKey(0)

def test_default_boxes():
    from utils.box import generate_anchors,generate_default_boxes
    from cv2 import cv2
    import numpy as np

    effective_stride = 16
    h,w = (600,600)
    fmap_h,fmap_w = int(h/effective_stride), int(w/effective_stride)

    anchors = generate_anchors(effective_stride)
    boxes = generate_default_boxes(anchors, (fmap_h,fmap_w), effective_stride)

    render(boxes)


def tt(anchors):
    import numpy as np
    import torch
    h,w = (600,600)
    fmap_h,fmap_w = int(h/16), int(w/16)
    shift_x = np.arange(0, fmap_w) * 16
    shift_y = np.arange(0, fmap_h) * 16
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                shift_x.ravel(), shift_y.ravel())).transpose())
    shifts = shifts.contiguous().float()

    A = 9
    K = shifts.size(0)

    all_anchors = anchors.view(1, A, 4) + shifts.view(K, 1, 4)
    all_anchors = all_anchors.reshape(K * A, 4)
    render(all_anchors)

def render(boxes):
    from cv2 import cv2
    import numpy as np

    blank_white = np.ones((600,600,3), dtype=np.uint8) * 255
    for x1,y1,x2,y2 in boxes.long().numpy():
        t_img = cv2.rectangle(blank_white.copy(), (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.imshow("",t_img)
        cv2.waitKey(2)

if __name__ == "__main__":
    from utils.box import generate_anchors
    tt(generate_anchors(16))
    #test_default_boxes()