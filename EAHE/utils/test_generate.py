import random as rd
import numpy as np
import cv2
import PIL.Image as Image
def cross(a, b, c):
    ans = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    return ans
def checkShape(a, b, c, d):
    x1 = cross(a, b, c)
    x2 = cross(b, c, d)
    x3 = cross(c, d, a)
    x4 = cross(d, a, b)

    if (x1 < 0 and x2 < 0 and x3 < 0 and x4 < 0) or (x1 > 0 and x2 > 0 and x3 > 0 and x4 > 0):
        return 1
    else:
        print('not convex')
        return 0

def generate_dataset(source_img):
    # define parameters
    # src_input1 = np.empty([4, 2], dtype=np.uint8)
    rho = 8.
    height = 128
    width = 128

    src_input1 = np.zeros([4, 2])
    # src_input2 = np.zeros([4, 2])
    dst = np.zeros([4, 2])
    # Upper left
    src_input1[0][0] = 0
    src_input1[0][1] = 0
    # Upper right
    src_input1[1][0] = width
    src_input1[1][1] = 0
    # Lower left
    src_input1[2][0] = 0
    src_input1[2][1] = height
    # Lower right
    src_input1[3][0] = width
    src_input1[3][1] = height
    # print(src_input1)

    offset = np.empty(8, dtype=np.int8)
    while True:
        for j in range(8):
            offset[j] = rd.randint(-rho, rho)
        # Upper left
        dst[0][0] = src_input1[0][0] + offset[0]
        dst[0][1] = src_input1[0][1] + offset[1]
        # Upper righ
        dst[1][0] = src_input1[1][0] + offset[2]
        dst[1][1] = src_input1[1][1] + offset[3]
        # Lower left
        dst[2][0] = src_input1[2][0] + offset[4]
        dst[2][1] = src_input1[2][1] + offset[5]
        # Lower right
        dst[3][0] = src_input1[3][0] + offset[6]
        dst[3][1] = src_input1[3][1] + offset[7]
        # print(dst)
        if checkShape(dst[0], dst[1], dst[3], dst[2]) == 1:
            break

    h, status = cv2.findHomography(dst, src_input1)

    # cv2.imwrite(r'/old_sharefiles/fengxiaomei/UnH_second_final/generate_data/img_warped.jpg',img_warped)
    # cv2.imwrite(r'/old_sharefiles/fengxiaomei/UnH_second_final/generate_data/img_warped_mask.jpg', img_warped_mask)
    # cv2.imwrite(r'/old_sharefiles/fengxiaomei/UnH_second_final/generate_data/img_warped.jpg', img_warped)

    # return offset, source_img, img_warped, img_warped_mask
    return offset, h

# print("Testing dataset...")
# # dataset_size = 1106
# generate_image_path = r'/fastersharefiles/fengxiaomei/homography_dataset/data_udis_test/'
# source_img = cv2.imread('/old_sharefiles/fengxiaomei/testing/left/000001.jpg')
# # source_img = Image.open('/old_sharefiles/fengxiaomei/testing/left/000001.jpg')
# source_img = cv2.resize(source_img,(128,128), interpolation=cv2.INTER_CUBIC)
# # source_img.resize((128, 128), Image.LANCZOS)
# # source_img = np.array(source_img)
# # source_img.resize((128, 128))
# generate_dataset(source_img)