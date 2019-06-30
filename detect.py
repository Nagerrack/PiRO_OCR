import main
import model as md
import nn_data_parser
import numpy as np
from scipy.stats import entropy
from plot_analysis import analyze
import copy as cp


# import matplotlib.pyplot as plt

def detect(path_to_image, weight_path):
    model = md.get_model()

    model.load_weights(weight_path)
    indice_images, wordsContours, modeWords = main.get_indices(path_to_image=path_to_image)

    entrops = []
    certainty = []
    drawns = []
    widths = []
    # indice_images.pop(0)
    for index in indice_images:
        # plt.imshow(index, cmap='gray')
        # plt.show()

        windows, width = nn_data_parser.slide_window(index)
        widths.append(width)
        prediction_list = []
        for window in windows:
            new_window = np.array(window, dtype=np.float32)
            new_window /= 255.0

            new_window = np.expand_dims(new_window, axis=-1)
            new_window = np.expand_dims(new_window, axis=0)

            # prediction_list.append(model.predict(new_window))
            prediction = model.predict(new_window)[0]
            # print(entropy(prediction))
            # print(prediction)
            prediction_list.append(prediction)
            # plt.imshow(window, cmap='gray')
            # plt.show()
            # print()
            # break

        vals = [np.amax(pred) for pred in prediction_list]
        # steps = [4 * ind for ind, pred in enumerate(prediction_list)]
        maxes = [np.argmax(pred) for pred in prediction_list]

        # valsTemp = [val for ind, val in enumerate(vals) if maxes[ind] != 10]
        ind = np.argpartition(vals, -12)[-12:]
        highest = np.array(vals)[ind]
        certainty.append(np.mean(highest))
        # print(np.mean(highest))
        total_entr = np.mean([entropy(pred) for ind, pred in enumerate(prediction_list) if maxes[ind] != 10])
        # print(entropy(maxes))
        entrops.append(total_entr)

        drawn, decisions = analyze(cp.deepcopy(vals), maxes)
        drawns.append((drawn))
        # for i in range(len(vals)):
        #     plt.annotate("{0}".format(maxes[i]), (steps[i], vals[i]))
        # plt.scatter(steps, vals, c=decisions)
        # plt.plot(steps, vals)
        # plt.show()
        # print(prediction_list)

    avgEntr = np.mean(entrops)
    avgCertainty = np.mean(certainty)
    # avgWidth = np.mean(widths)
    # print(avgEntr)
    # print(avgCertainty)
    # print(entrops[0])
    # print(certainty[0])
    # print(avgWidth)
    # print(widths[0])
    multip = 1
    avgLineWidth = np.mean([wordsCont[0][1][0][1] for wordsCont in wordsContours])
    lineWidth = wordsContours[0][0][1][0][1]
    # print(lineWidth)
    # print(avgLineWidth)
    lineWidthRatio = abs(lineWidth - avgLineWidth) / avgLineWidth
    # if lineWidthRatio < 0.16:
    #     multip = 0.965

    # if len(drawns[0])<5:
    #     multip +=0.017
    #     multip+= 0.003 * (5-len(drawns[0]))
    if avgCertainty * 0.955 * multip > certainty[0] and avgEntr * 1.16 / multip < entrops[
        0] or avgCertainty * 0.91 * multip > certainty[0] or avgEntr * 1.32 / multip < entrops[0] \
            or lineWidthRatio > 0.29:
        drawns.pop(0)
    return drawns


def check():
    model = md.get_model()

    weight_path = 'weights/weightsAvg4'

    model.load_weights(weight_path)
    for numb in range(1, 30):
        indice_images, wordsContours, modeWords = main.get_indices(numb)

        # c = 0

        entrops = []
        certainty = []
        drawns = []
        widths = []
        # indice_images.pop(0)
        for index in indice_images:
            # plt.imshow(index, cmap='gray')
            # plt.show()

            windows, width = nn_data_parser.slide_window(index)
            widths.append(width)
            prediction_list = []
            for window in windows:
                new_window = np.array(window, dtype=np.float32)
                new_window /= 255.0

                new_window = np.expand_dims(new_window, axis=-1)
                new_window = np.expand_dims(new_window, axis=0)

                # prediction_list.append(model.predict(new_window))
                prediction = model.predict(new_window)[0]
                # print(entropy(prediction))
                # print(prediction)
                prediction_list.append(prediction)
                # plt.imshow(window, cmap='gray')
                # plt.show()
                # print()
                # break

            vals = [np.amax(pred) for pred in prediction_list]
            steps = [4 * ind for ind, pred in enumerate(prediction_list)]
            maxes = [np.argmax(pred) for pred in prediction_list]

            valsTemp = [val for ind, val in enumerate(vals) if maxes[ind] != 10]
            ind = np.argpartition(vals, -12)[-12:]
            highest = np.array(vals)[ind]
            certainty.append(np.mean(highest))
            # print(np.mean(highest))
            total_entr = np.mean([entropy(pred) for ind, pred in enumerate(prediction_list) if maxes[ind] != 10])
            # print(entropy(maxes))
            entrops.append(total_entr)

            drawn, decisions = analyze(cp.deepcopy(vals), maxes)
            drawns.append((drawn))
            # for i in range(len(vals)):
            #     plt.annotate("{0}".format(maxes[i]), (steps[i], vals[i]))
            # plt.scatter(steps, vals, c=decisions)
            # plt.plot(steps, vals)
            # plt.show()
            # print(prediction_list)

        avgEntr = np.mean(entrops)
        avgCertainty = np.mean(certainty)
        # avgWidth = np.mean(widths)
        # print(avgEntr)
        # print(avgCertainty)
        # print(entrops[0])
        # print(certainty[0])
        # print(avgWidth)
        # print(widths[0])
        multip = 1
        avgLineWidth = np.mean([wordsCont[0][1][0][1] for wordsCont in wordsContours])
        lineWidth = wordsContours[0][0][1][0][1]
        print(lineWidth)
        print(avgLineWidth)
        lineWidthRatio = abs(lineWidth - avgLineWidth) / avgLineWidth
        # if lineWidthRatio < 0.16:
        #     multip = 0.965

        # if len(drawns[0])<5:
        #     multip +=0.017
        #     multip+= 0.003 * (5-len(drawns[0]))
        if avgCertainty * 0.955 * multip > certainty[0] and avgEntr * 1.16 / multip < entrops[
            0] or avgCertainty * 0.91 * multip > certainty[0] or avgEntr * 1.32 / multip < entrops[0] \
                or lineWidthRatio > 0.29:
            print(str(numb) + " remove first")
