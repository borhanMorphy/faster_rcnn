import sys
from cv2 import cv2
import selectivesearch

def scale_img(img, height:int=200):
    sf = height/img.shape[0]
    width = int(img.shape[1]*sf)
    return cv2.resize(img, (width,height)),sf


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])

    #img,sf = scale_img(img)
    sf = 1
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    print("found: ",len(regions))
    for region in regions:
        x1,y1,x2,y2 = [int(b/sf) for b in region['rect']]
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)

    cv2.imshow("",img)
    cv2.waitKey(0)