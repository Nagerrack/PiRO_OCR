import numpy as np

from Text_Area_Detector import show, detect_one_img
from image_io import *
from surrounding_removal import remove_surrounding


class Piromain():

    def remove_background(self, img):

        can = cv2.Canny(img, 100, 200, None, 3)
        lines = cv2.HoughLinesP(can, 1, np.pi / 180, 180, None, 100, 50)
        cdst = np.zeros(shape=np.shape(img))
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

        img = np.uint8(can - cdst)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       self.adaptiveThreshBlockSize, self.adaptiveThreshholdAddVal)
        return thresh

    def detect_lines(self, img, thresh):
        thresh = cv2.bitwise_not(thresh)
        kernel = np.array([[1] * self.kernelLen])
        thresh = cv2.filter2D(thresh, 5, kernel)
        thresh = np.uint8(thresh)

        thresh = cv2.erode(thresh, kernel=np.ones(shape=(3, 1)))
        thresh = cv2.bitwise_not(thresh)

        # suma w wierszach i przeskalowanie zakresu wartości
        integral = np.sum(thresh, axis=1)
        integral = integral / np.max(integral) * 255

        # utworzenie obrazu o oryginalnym rozmiarze
        rep = np.transpose(np.tile(integral, (img.shape[1], 1)))
        rep = np.uint8(rep / np.max(rep) * 255)

        return rep

    def broaden_lines(self, rep):
        # progowanie w punkcie średniej obrazu i rozszerzenie obszarów białych
        ret, rep = cv2.threshold(rep, np.mean(rep), 255, cv2.THRESH_BINARY)
        rep = cv2.erode(rep, kernel=np.ones(shape=(3, 1)))
        return rep

    def findBorderLengthsOfContour(self, contours, ind, img):
        height, width = np.shape(img)

        if ind == 0:
            borderDown = height - contours[ind][1][0][1]
        else:
            borderDown = contours[ind - 1][0][0][1] - contours[ind][1][0][1]
        if ind == len(contours) - 1:
            borderUp = contours[ind][0][0][1]
        else:
            borderUp = contours[ind][0][0][1] - contours[ind + 1][1][0][1]
        return borderUp * width, borderDown * width

    def detectContours(self, img):
        im2, contoursB, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img = cv2.bitwise_not(img)
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # delete 2 rows overlapping
        contourAres = [cv2.contourArea(cont) for cont in contours]
        area = np.mean(np.sort(contourAres)[int(0.25 * len(contourAres)):int(np.ceil(0.75 * len(contourAres)))])

        i = 0
        for cont in contoursB:
            ar = cv2.contourArea(cont)
            borderUp, borderDown = self.findBorderLengthsOfContour(contoursB, i, img)
            if (ar < 0.4 * area and (borderUp < 0.6 * area or borderDown < 0.6 * area)):
                cv2.drawContours(img, contoursB, i, (255, 255, 255), cv2.FILLED)
            i += 1
        # show(img)
        for cont in contours:
            for i in [4, 3, 2]:
                if cv2.contourArea(cont) > i * area:
                    M = cv2.moments(cont)
                    cy = int(M['m01'] / M['m00'])
                    upper = cont[0][0][1]
                    lower = cont[1][0][1]
                    size = lower - upper
                    if i % 2 == 0:
                        img[cy, :] = 0
                    if i % 2 != 0:
                        img[cy + int(size / 6), :] = 0
                        img[cy - int(size / 6), :] = 0
                    if i % 4 == 0:
                        img[cy + int(size / 4), :] = 0
                        img[cy - int(size / 4), :] = 0
                    break
        ########3
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # show(img)
        i = 0
        todel = []
        for cont in contours:
            temp = np.zeros(shape=np.shape(img))
            cv2.drawContours(temp, contours, i, (255, 255, 255), thickness=5)
            # show(img, temp)
            M = cv2.moments(cont)
            bad = False
            ar = cv2.contourArea(cont)
            # print(ar)
            borderUp, borderDown = self.findBorderLengthsOfContour(contours, i, img)
            if (ar < 0.3 * area and (borderUp < 0.6 * area or borderDown < 0.6 * area)) or (
                    borderUp + borderDown < area * 0.5) or (
                    i == 0 and borderUp > 2 * area and borderDown * 1.3 < borderUp) \
                    or (i == len(contours) - 1 and borderDown > 2 * area and borderUp * 1.3 < borderDown):
                bad = True
            for coord in cont:
                if coord[0][1] == 0 or coord[0][1] == np.shape(img)[0]:
                    bad = True
            if M['m00'] != 0 and not bad:
                cy = int(M['m01'] / M['m00'])
                cv2.line(img, (0, cy), (np.shape(img)[1], cy), (127, 127, 127), 21)
            else:
                todel.append(i)
            i += 1

        contours = np.delete(contours, todel, axis=0)
        print("Wykrytych wierszy: " + str(len(contours)))
        show(img)
        return img

    def apply_mask(self, rep, img):
        return np.where(rep == 127, img, 0)

    def get_global_params(self, img):
        # params on color
        self.height, self.width, self.channels = np.shape(img)
        self.param = np.sqrt(self.height ** 2 + self.width ** 2)

        # params on gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.mean = np.mean(img)

        self.filterMask = 5
        self.adaptiveThreshBlockSize = int(self.param / 100) + 1 + int(self.param / 100) % 2
        self.adaptiveThreshholdAddVal = 7 + int(self.mean / 10)
        if self.param > 800:
            self.filterMask += 2
        edges = cv2.Canny(img, self.mean / 2, self.mean)

        self.edgesMean = np.mean(edges)
        self.erode1 = 3 + int(self.param / 600) + int(self.param / 600) % 2
        self.dilateFirst = 3 + int(self.param / 400) * 2
        self.erodeFirst = self.dilateFirst + int(self.param / 600) * 2
        self.kernelLen = 3 + int(self.param / 800) * 2

    def swap_channel(self, img):

        temp = np.copy(img[:, :, 0])
        temp2 = np.copy(img[:, :, 2])

        img[:, :, 2] = temp
        img[:, :, 0] = temp2
        return img

    def main(self):
        for i in range(1, 30):
            imgs = read_particular_images("data", [i])
            img = self.swap_channel(imgs[0])
            img = remove_surrounding(img)
            show(img)
            print(i)
            self.get_global_params(img)

            img, box, M = detect_one_img(img, True)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=np.mean(img))

            show(img)
            thresh = self.remove_background(img)
            rep = self.detect_lines(img, thresh)
            rep = self.broaden_lines(rep)
            rep = self.detectContours(rep)
            result = self.apply_mask(rep, img)
            show(result)
            # display_image(result, i)


if __name__ == "__main__":
    m = Piromain()
    m.main()
