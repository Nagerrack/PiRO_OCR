from detect import detect
from pprint import pprint
def ocr(path_to_img):
    weight_path = 'weights/weightsAvgV3-6_original_nodrop_Final'
    drawns = detect(path_to_img, weight_path)
    result = [('', '', ''.join([str(num) for num in drawn])) for drawn in drawns]
    # pprint(result)
    # print("-----------------------------")
    # weight_path = 'weights/weightsAvg4'
    # drawns = detect(path_to_img, weight_path)
    # result = [('', '', ''.join([str(num) for num in drawn])) for drawn in drawns]
    # pprint(result)
    # print("-----------------------------")

    return result

if __name__ == "__main__":
    ocr('data/img_1.jpg')