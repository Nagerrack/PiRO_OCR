import numpy as np
import cv2
import matplotlib.pyplot as plt


def slide_window(img, step_size: int = 4, w_height: int = 48, w_width: int = 32, create_image_with_part_step=True):
    h = img.shape[0]

    assert h > w_height, "Image height is less than window height"


    vccp = (h - w_height) // 2  # vertical_crop_corner_position

    img = img[vccp:vccp + w_height,:]
    #imgT = np.transpose(img)

    addImg = cv2.imread('kratki_extracted/0.png', 0)
    img = np.concatenate([addImg, img, addImg], axis=1)

    w = img.shape[1]
    assert w > w_width, "Image width is less than window width"
    assert step_size > 1, "Step size must be grater than 1"

    itr_number = (w - w_width) // step_size + 1
    missing_full_step_pixels = (w - w_width) % step_size
    result = []
    for i in range(itr_number):
        hccp = i * step_size
        crop_img = img[:, hccp:hccp + w_width]
        result.append(crop_img)

    if create_image_with_part_step and missing_full_step_pixels > 0:
        crop_img = img[:, w - w_width:w]
        result.append(crop_img)
    return result


def aggregate_output_in_index(nn_outputs):
    divided_nn_output = divide(nn_outputs)
    return aggregate(divided_nn_output)


def divide(nn_outputs, minimum_sample_per_digit=1):
    accumulator_index = []
    accumulator_digit = []
    # current_max_index = 11
    for i in range(len(nn_outputs)):
        max_index = np.argmax(nn_outputs[i])
        # here add other condition separation
        if max_index != 10:
            accumulator_digit.append(nn_outputs[i])
        else:
            if len(accumulator_digit) >= minimum_sample_per_digit:
                accumulator_index.append(accumulator_digit)
                accumulator_digit = []

    # boundary condition
    if len(accumulator_digit) >= minimum_sample_per_digit:
        accumulator_index.append(accumulator_digit)
    return accumulator_index


def aggregate(divided_nn_outputs):
    result = ""
    for element in divided_nn_outputs:
        sum_array = np.sum(element, axis=0)
        index = np.argmax(sum_array)
        result = result + str(index)
    return result


class LastMinuteMemory:

    def __init__(self, step_to_remember=3):
        self.memory = []
        self.step_to_remember = step_to_remember

    def remember(self, item):
        self.memory.append(item)
        forgotten_value = self.forget_distant_past()
        return forgotten_value

    def forget_distant_past(self):
        if (len(self.memory) > self.step_to_remember):
            return self.memory.pop(0)
        return None

    def forget_all(self):
        self.memory.clear()


class IndexAggregator():
    def __init__(self, nn_outputs, accept_probability_threshold=0.80, ignore_the_same_value=3):
        self.nn_outputs = nn_outputs
        self.memory = LastMinuteMemory(ignore_the_same_value)
        self.accept_probability_threshold = accept_probability_threshold
        self.index = ''
        self.barrier = 10

    def aggregate(self):
        for nn_output in self.nn_outputs:
            self.get_decision(nn_output)
        return self.index

    def get_decision(self, nn_output):
        pretendendt = np.argmax(nn_output)

        if np.max(nn_output) > self.accept_probability_threshold:
            if not self.is_block(pretendendt):
                self.index += str(pretendendt)
                self.blocked_value = pretendendt

    def is_block(self, value):
        forgoten = self.memory.remember(value)
        if self.barrier != value:  # Block value is different than incoming
            if value == 10:
                self.barrier = 10
                return True
            if self.barrier == 8 and value == 3:  # RULE FOR EIGHT
                if 8 in self.memory.memory:
                    return True  # BLOCK
            self.barrier = value
            return False  # PASS
        else:  # The same value detected
            if forgoten == value:
                return True
            return True  # BLOCK


def test_method_1():
    img = cv2.imread("data/raw_index/1.png")
    imgs = slide_window(img, w_height=100, w_width=50)
    plt.imshow(img, cmap='gray')
    plt.show()
    for item in imgs:
        plt.imshow(item, cmap='gray')
        plt.show()


def est_method_2():
    array = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],  # NaN
             [0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0],  # 4
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],  # NaN
             [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],  # NaN
             [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
             [0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0],  # 8
             [0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0],  # 8
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],  # NaN
             [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],  # NaN
             ]

    print("hello World")
    # print(aggregate_output_in_index(array))
    a = IndexAggregator(array)
    print(a.aggregate())
if __name__ == "__main__":

    est_method_2()
