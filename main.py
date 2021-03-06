import numpy as np

from Text_Area_Detector import detect_one_img  # , show
from image_io import *
from surrounding_removal import remove_surrounding
# import copy as cp
from IndexWordDecisionProcess import decisionProcess


class Piromain():

    def remove_background(self, img):

        can = cv2.Canny(img, 100, 200, None, 3)
        lines = cv2.HoughLinesP(can, 1, np.pi / 180, 180, None, 100, 150)
        cdst = np.zeros(shape=np.shape(img))
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)
        ##show(can)
        img = np.uint8(can - cdst)
        ##show(img)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       self.adaptiveThreshBlockSize, self.adaptiveThreshholdAddVal)
        ##show(img)
        return thresh

    def detect_lines(self, img, thresh):
        ##show(thresh)
        thresh = cv2.bitwise_not(thresh)
        kernel = np.array([[1] * self.kernelLen])
        thresh = cv2.filter2D(thresh, 5, kernel)
        ##show(thresh)
        thresh = np.uint8(thresh)
        ##show(thresh)
        thresh = cv2.erode(thresh, kernel=np.ones(shape=(3, 1)))
        thresh = cv2.bitwise_not(thresh)
        ##show(thresh)
        # suma w wierszach i przeskalowanie zakresu wartości
        integral = np.sum(thresh, axis=1)
        integral = integral / np.max(integral) * 255

        # utworzenie obrazu o oryginalnym rozmiarze
        rep = np.transpose(np.tile(integral, (img.shape[1], 1)))
        rep = np.uint8(rep / np.max(rep) * 255)

        return rep

    def broaden_lines(self, rep):
        # progowanie w punkcie średniej obrazu i rozszerzenie obszarów białych
        ##show(rep)
        h, w = np.shape(rep)
        start = 0
        mean = np.mean(rep)

        #
        end = start + int(h / 8 * (1))

        ret, th = cv2.threshold(rep[start:end], np.mean(np.mean(rep[start:end])) * 0.98, 255, cv2.THRESH_BINARY)
        # print(np.mean(rep[start:end]))
        rep[start:end] = th
        start = end

        # print(start)
        ##show(rep)
        end = h - int(h / 8 * (1))
        ret, th = cv2.threshold(rep[start:end], np.mean(np.mean(rep[start:end])), 255, cv2.THRESH_BINARY)
        rep[start:end] = th
        ##show(rep)
        start = end
        end = h - 1
        ret, th = cv2.threshold(rep[start:end], np.mean(np.mean(rep[start:end])) * 0.98, 255, cv2.THRESH_BINARY)
        rep[start:end] = th
        ##show(rep)

        # ret, rep = cv2.threshold(rep, np.mean(rep), 255, cv2.THRESH_BINARY)
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

    def checkIfAtTheEnd(self, contours, i, img, threshVal=0):
        borderUp, borderDown = self.findBorderLengthsOfContour(contours, i, img)
        if borderUp < threshVal or borderDown < threshVal:
            return True
        return False

    def detectContoursForCols(self, img):
        im2, contoursB, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img = cv2.bitwise_not(img)
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # delete 2 rows overlapping
        contourAres = [cv2.contourArea(cont) for cont in contours]
        area = np.shape(img)[0] * np.shape(img)[1]
        h, w = np.shape(img)
        numb = 3
        if len(contours) > 1:
            ind = -1
            if contours[-1][0][0][1] == 0:
                ind = -2
            if contours[ind][0][0][1] > h * 0.01 and abs(
                    contours[ind][0][0][1] - contours[ind][1][0][1]) < h * 0.07 and abs(
                contours[ind][0][0][1] - contours[ind][1][0][1]) > h * 0.03 and contours[ind][1][0][1] < 0.1 * h:
                numb += 1
        if len(contours) > numb:
            i = 0
            filled = 0

            for cont in contoursB:
                ar = cv2.contourArea(cont)
                borderUp, borderDown = self.findBorderLengthsOfContour(contoursB, i, img)
                if (ar < 0.1 * area and borderUp < 0.125 * area and borderDown > 0.04 * area and i == 0) and len(
                        contours) - filled > numb:
                    cv2.drawContours(img, contoursB, i, (255, 255, 255), cv2.FILLED)
                    filled += 1
                    continue
                if ((i != 0 and i != len(contoursB) - 1) and ar < 0.1 * area and (
                        borderUp + borderDown < 0.15 * area or (
                        borderUp < 0.1 * area and borderDown < 0.1 * area))) and len(contours) - filled > numb:
                    cv2.drawContours(img, contoursB, i, (255, 255, 255), cv2.FILLED)
                    filled += 1
                    continue

                i += 1
            # show(img)
        ########3
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # #########show(img)
        i = 0
        todel = []

        for cont in contours:
            temp = np.zeros(shape=np.shape(img))
            cv2.drawContours(temp, contours, i, (255, 255, 255), thickness=5)

            M = cv2.moments(cont)
            bad = False
            ar = cv2.contourArea(cont)
            # print(ar)
            borderUp, borderDown = self.findBorderLengthsOfContour(contours, i, img)
            if len(contours) - len(todel) > numb:
                if ar < 0.04 * area and borderDown < 0.04 * area and borderUp > 0.1 * area and i == 0:
                    bad = True

                if ar < 0.04 * area and borderDown > 0.1 * area and borderDown < 0.04 * area and i == len(contours) - 1:
                    bad = True

                for coord in cont:
                    if coord[0][1] == 0 or (coord[0][1] == np.shape(img)[0] and len(contours) - len(todel) > 4):
                        bad = True
            if M['m00'] == 0 or bad:
                todel.append(i)
            i += 1
        contours = np.delete(contours, todel, axis=0)

        # show(img)
        if len(contours) > numb:
            img = cv2.bitwise_not(img)
            im2, contoursB, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areasB = {en: cv2.contourArea(cont) + sum(self.findBorderLengthsOfContour(contoursB, en, img)) for en, cont
                      in enumerate(contoursB) if not self.checkIfAtTheEnd(contoursB, en, img, w * 4)}
            deleted = 0
            borders = list(areasB.items())
            while (len(contours) - deleted > numb):
                minBord = min(borders, key=lambda x: x[1])
                cv2.drawContours(img, contoursB, minBord[0], (127, 127, 127), cv2.FILLED)
                borders.remove(minBord)
                deleted += 1
                # show(img)
            img = cv2.bitwise_not(img)
        numb = len(contours)
        for i in range(len(contours)):
            cv2.drawContours(img, contours, i, (127, 127, 127), cv2.FILLED)

            # show(img)

        # print("Wykrytych wierszy: " + str(len(contours)))
        ##########show(img)
        return img, numb, contours

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
                ##show(img)
            i += 1
        # #########show(img)
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
        ##show(img)
        i = 0
        todel = []
        heights = []
        for cont in contours:
            temp = np.zeros(shape=np.shape(img))
            cv2.drawContours(temp, contours, i, (255, 255, 255), thickness=5)
            ##show(img, temp)
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
                heights.append(cy)
                cv2.line(img, (0, cy), (np.shape(img)[1], cy), (127, 127, 127), 1)
            else:
                todel.append(i)
            i += 1

        contours = np.delete(contours, todel, axis=0)
        # print("Wykrytych wierszy: " + str(len(contours)))
        ##########show(img)
        return img, len(contours), heights

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
        edges = cv2.Canny(img, self.mean, self.mean * 2)
        # show(edges)
        self.edgesMean = np.mean(edges)
        self.edgesVar = np.var(edges)
        self.erode1 = 3 + int(self.param / 600) + int(self.param / 600) % 2
        self.dilateFirst = 3 + int(self.param / 400) * 2
        self.erodeFirst = self.dilateFirst + int(self.param / 600) * 2
        self.kernelLen = 3 + int(self.param / 800) * 2

        # print(self.edgesMean)
        # print(self.edgesVar)
        # print(self.mean)
        # print(np.sqrt(self.param))

    def swap_channel(self, img):

        temp = np.copy(img[:, :, 0])
        temp2 = np.copy(img[:, :, 2])

        img[:, :, 2] = temp
        img[:, :, 0] = temp2
        return img

    def detectWords(self, row):

        # show(row)
        row = np.transpose(row)
        # show(row)
        thresh = self.remove_background(row)
        # print("now")
        kernel = np.ones(shape=(5, 5))
        thresh = cv2.erode(thresh, kernel, iterations=3)

        # thresh = cv2.bitwise_not(thresh)

        # show(thresh)
        rep = self.detect_lines(row, thresh)
        # show(rep)
        rep = self.broaden_lines(rep)
        # show(rep)

        rep = cv2.dilate(rep, kernel, iterations=8)
        # show(rep)
        rep = cv2.erode(rep, kernel, iterations=8)
        # show(rep)
        rep, number, contours = self.detectContoursForCols(rep)
        # show(rep)
        rep = np.where(rep != 127, 0, rep)
        rowsX = rep[:, 0]
        return rowsX, number, contours

        result = self.apply_mask(rep, row)
        ###show(result)

    def itret(self, n, val, ls, dir):

        if n + dir < len(ls) and n + dir >= 0 and (
                ls[n + dir] == val + dir - 10 or abs(ls[n + dir] - (val + dir)) < 100):
            return self.itret(n + dir, val + dir, ls, dir)
        return n

    def mode(self, collection):
        (_, idx, counts) = np.unique(collection, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode = collection[index]
        return mode

    def mainLoop(self):
        # with open('data/iloscWierszy.txt', 'r') as file:
        #     lines=file.readlines()
        # lines.pop(0)
        # rowsCheck=[int(line.split(',')[0]) for line in lines]
        # print(rowsCheck)
        # rowsList = []
        for i in range(1, 30):
            # i+=15
            indexesImgs = self.main(i)
            for indImg in indexesImgs:
                pass
                # plt.imshow(indImg)
                # plt.show()

    def main(self, i=None, path_to_image=None):
        # i+=12
        if path_to_image is None:

            img = read_particular_images("data", [i])[0]
        else:
            img = cv2.imread(path_to_image)
        # imgOrg = cp.deepcopy(imgs[0])
        # print(imgOrg.shape)
        img = self.swap_channel(img)
        ##show(img)
        orgRows, orgCols, chan = np.shape(img)
        img, deleted = remove_surrounding(img)
        # fromUp = self.itret(-1, -1, deleted['rows'], 1) + 1
        # fromDown = len(deleted['rows']) - self.itret(len(deleted['rows']), orgRows, deleted['rows'], -1)
        # fromLeft = self.itret(-1, -1, deleted['columns'], 1) + 1
        # fromRight = len(deleted['columns']) - self.itret(len(deleted['columns']), orgCols, deleted['columns'], -1)

        ##show(img)
        # print(i)
        self.get_global_params(img)

        img, box, angle, pointsCut = detect_one_img(img, True)
        # print(pointsCut)
        # show(img)
        # show(imgOrg)
        # imgWork = imgOrg[pointsCut[1]+50:pointsCut[1]+250, 200:pointsCut[3]]
        # show(imgWork)
        ##show(img)
        rowsOfImg, colsOfImg = img.shape
        M = cv2.getRotationMatrix2D((colsOfImg / 2, rowsOfImg / 2), angle, 1)
        # MT = cv2.getRotationMatrix2D((colsOfImg / 2, rowsOfImg / 2), -angle, 1)
        # print(M)
        ##show(img)
        # img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=np.mean(img))

        ##show(img)

        ##show(img)
        thresh = self.remove_background(img)
        rep = self.detect_lines(img, thresh)
        rep = self.broaden_lines(rep)
        rep, rows, heights = self.detectContours(rep)

        # heights.pop()
        # heights.pop(0)
        # for ind,height in enumerate(heights):
        #     tempImg = imgOrg[height-30:height+30, pointsCut[3]+150:np.shape(imgOrg)[1]-50]
        #     cv2.imwrite('kratki/'+str(i) + '_' +str(ind)+'.jpg',tempImg)
        #     #plt.imshow(tempImg)
        #     #plt.show()

        ##show(rep)
        result = self.apply_mask(rep, img)
        ##show(result)
        # rowsList.append(rows)
        rep = np.where(rep != 127, 0, rep)
        ##show(rep)
        x = np.where(rep == 127)
        rowsY = np.unique(x[0])
        # print(rowsY)
        retMat = np.zeros(shape=np.shape(img))
        wordsNumbs = []
        wordsContsList = []
        for k in range(len(rowsY)):
            rowsX, wordsNumber, wordsConts = self.detectWords(img[rowsY[k] - 30:rowsY[k] + 30])
            wordsNumbs.append(wordsNumber)
            wordsContsList.append(wordsConts)
            retMat[rowsY[k]] = np.where(rowsX > 0, k + 1, 0)

        # ker = np.ones(shape=(11, 1))
        # retMat = cv2.dilate(retMat, ker)
        # ##show(retMat)
        # retMat = cv2.warpAffine(retMat, MT, (img.shape[1], img.shape[0]), borderValue=0, flags=cv2.INTER_NEAREST)
        # retMat2 = np.zeros(shape=(orgRows, orgCols))
        # retMat2[pointsCut[0] + fromUp:pointsCut[1] + fromUp,
        # pointsCut[2] + fromLeft:pointsCut[3] + fromLeft] = retMat
        # # print(retMat2.shape)
        # ###show(retMat2)
        # # imgOrg[:,:,2] = retMat2
        # # show(imgOrg)
        # ###show(result)
        # # display_image(result, i)
        # plt.imshow(retMat2)  ### DO ZAKOMENTOWANIA WYSWIETLANIE
        # plt.show()  ### DO ZAKOMENTOWANIA WYSWIETLANIE

        # cv2.namedWindow('test')
        #
        #
        # global noLinesImg
        # noLinesImg = cp.deepcopy(img)
        # global imgSaved
        # imgSaved = cp.deepcopy(img)
        # global x1, x2,x3,x4,x5
        #
        # x1=168
        # x2=212
        # x3=144
        # x4=274
        # x5=216
        # # if self.edgesMean < 10:
        # #     x4 = 45
        # #     x5 = 182
        # def change1(x):
        #     global x1
        #     x1= x
        #
        # def change2(x):
        #     global x2
        #     x2 = x
        #
        # def change3(x):
        #     global x3
        #     x3 = x
        # def change4(x):
        #     global x4
        #     x4 = x
        # def change5(x):
        #     global x5
        #     x5 = x
        #
        # cv2.createTrackbar('1', 'test', x1, 500, change1)
        # cv2.createTrackbar('2', 'test', x2, 500, change2)
        # cv2.createTrackbar('3', 'test', x3, 500, change3)
        # cv2.createTrackbar('4', 'test', x4, 500, change4)
        # cv2.createTrackbar('5', 'test', x5, 500, change5)
        # while cv2.waitKey(30) != ord('q'):
        #     can = cv2.Canny(imgSaved, x4, x5, None, 3)
        #     lines = cv2.HoughLinesP(can, 1, np.pi / 180, x1, None, x2, x3)
        #     cdst = np.zeros(shape=np.shape(imgSaved))
        #     if lines is not None:
        #         for i in range(0, len(lines)):
        #             l = lines[i][0]
        #             cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)
        #     # show(can)
        #     noLinesImg = np.uint8(can - cdst)
        #     noLinesImg = cv2.resize(noLinesImg, (0, 0), fx=0.5, fy=0.5)
        #     # show(img)
        #     cv2.imshow('test', noLinesImg)

        # lines = cv2.HoughLinesP(can, 1, np.pi / 180, 180, None, 100, 150)
        # cdst = np.zeros(shape=np.shape(img))
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         l = lines[i][0]
        #         cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)
        # #show(can)
        # img = np.uint8(can - cdst)
        # #show(img)

        modeWords = self.mode(wordsNumbs)
        if wordsNumbs.count(3) / len(wordsNumbs) > 0.35:
            modeWords = 3
        modeIndices = np.where(np.array(wordsNumbs) == modeWords)[0]
        modeGroupMean = np.mean(
            [abs(wordsContsList[ind][0][1][0][1] - wordsContsList[ind][0][0][0][1]) for ind in modeIndices])

        # print(modeGroupMean)
        if modeGroupMean < 100 or modeGroupMean > 280:
            modeGroupMean = np.sqrt(self.param) * 3.2
        modeGroupBorderMean = np.mean([abs(
            wordsContsList[ind][0][0][0][1] - (wordsContsList[ind][1][1][0][1] if len(wordsContsList[ind]) > 1 else 0))
            for ind in modeIndices])
        # print(modeGroupBorderMean)

        if modeGroupBorderMean < 30 or modeGroupBorderMean > 125:
            modeGroupBorderMean = np.sqrt(self.param) * 1.2
        allDistances = [abs(
            wordsContsList[ind][0][0][0][1] - (wordsContsList[ind][1][1][0][1] if len(wordsContsList[ind]) > 1 else 0))
            for ind, numb in enumerate(wordsNumbs)]
        allSizes = [abs(wordsContsList[ind][0][1][0][1] - wordsContsList[ind][0][0][0][1]) for ind, numb in
                    enumerate(wordsNumbs)]
        allBeforeSizes = [abs(wordsContsList[ind][1][1][0][1] - wordsContsList[ind][1][0][0][1]) if len(
            wordsContsList[ind]) > 1 else 0 for ind, numb in enumerate(wordsNumbs)]

        dataAll = [(allDistances[ind],
                    numb,
                    allSizes[ind],
                    allBeforeSizes[ind] + allDistances[ind] + allSizes[ind])
                   for ind, numb in enumerate(wordsNumbs)]
        params = [modeWords, modeGroupMean, modeGroupBorderMean]

        images = []
        for ind, data in enumerate(dataAll):
            state = 'none'
            while (state != 'accept' and len(wordsContsList[ind]) > 1):
                if state == 'remove':
                    wordsContsList[ind] = np.delete(wordsContsList[ind], 0, axis=0)
                    dist = abs(wordsContsList[ind][0][0][0][1] - (
                        wordsContsList[ind][1][1][0][1] if len(wordsContsList[ind]) > 1 else 0))
                    size = abs(wordsContsList[ind][0][1][0][1] - wordsContsList[ind][0][0][0][1])
                    before_size = abs(wordsContsList[ind][1][1][0][1] - wordsContsList[ind][1][0][0][1]) if len(
                        wordsContsList[ind]) > 1 else 0
                    data = (dist, data[1] - 1, size, before_size + dist + size)
                if state == 'merge':
                    popped = wordsContsList[ind][0]
                    wordsContsList[ind] = np.delete(wordsContsList[ind], 0, axis=0)
                    wordsContsList[ind][0][1][0][1] = popped[1][0][1]
                    dist = abs(wordsContsList[ind][0][0][0][1] - (
                        wordsContsList[ind][1][1][0][1] if len(wordsContsList[ind]) > 1 else 0))
                    size = abs(wordsContsList[ind][0][1][0][1] - wordsContsList[ind][0][0][0][1])
                    before_size = abs(wordsContsList[ind][1][1][0][1] - wordsContsList[ind][1][0][0][1]) if len(
                        wordsContsList[ind]) > 1 else 0
                    data = (dist, data[1] - 1, size, before_size + dist + size)
                state = decisionProcess(data, params)
                # print(state)

            # imgIndex = img[rowsY[ind] - 30:rowsY[ind] + 30,
            #            wordsContsList[ind][0][0][0][1] - int(0.9 * data[0]) + 1:wordsContsList[ind][0][1][0][1] + int(
            #                0.9 * data[0]) + 1]
            imgIndex = img[rowsY[ind] - 30:rowsY[ind] + 30,
                       wordsContsList[ind][0][0][0][1]:wordsContsList[ind][0][1][0][
                           1]]
            # imgIndex = img[rowsY[ind]-30:rowsY[ind]+30, wordsContsList[ind][0][1][0][1]+int(0.9*data[0])+1:wordsContsList[ind][0][1][0][1]+int(0.9*data[0])+100]
            images.append(imgIndex)
            # print(wordsContsList)
            # plt.imshow(imgIndex)
            # plt.show()

        ## UWAGA! Pierwszy INDEX to zazwyczaj NIE INDEX TYLKO OSTATNI WYRAZ Z NAGŁÓWKA LISTY -> przekazuje dalej
        # do odrzucenia tego case'u !
        return images, wordsContsList, modeWords  ### TO TRZEBA RETURNOWAC TYLKO INACZEJ PĘTLE TRZEBA OBUDOWAC <-
    # err = 0
    # errList=[]
    # for i in range(len(rowsCheck)):
    #
    #     if rowsCheck[i] != rowsList[i]:
    #         err+=1
    #         errList.append(i+1)
    # print(err)
    # print(errList)


def get_indices(number_of_image=None, path_to_image=None):
    m = Piromain()
    return m.main(number_of_image, path_to_image)


if __name__ == "__main__":
    m = Piromain()
    m.mainLoop()
