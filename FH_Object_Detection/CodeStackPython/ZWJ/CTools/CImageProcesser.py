import os
import sys
import cv2

class CImageProcesser:
    def img_resize(self,img_file,resize_resolution,bool_vis=False,bool_output=False):
        img = cv2.imread(img_file)
        resize_img = cv2.resize(img,resize_resolution,interpolation=cv2.INTER_AREA)

        if bool_vis:
            cv2.imshow("original img",img)
            cv2.waitKey(-1)
            cv2.destroyAllWindows()
            cv2.imshow("resized img",resize_img)
            cv2.waitKey(-1)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = '/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/2022-01-12-16-48-31/Image/1641977313231.png'
    processer = CImageProcesser()

    processer.img_resize(image_path,(1280,1024),True)
