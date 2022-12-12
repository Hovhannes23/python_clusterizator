import base64
import io
import os
from base64 import encodebytes

from clustimage import Clustimage
from sklearn.cluster import MiniBatchKMeans
from utils import *
import preprocess


def get_cells_from_image(img, clusters_num, rows_num, columns_num):
    clusters_num = int(clusters_num)
    rows_num = int(rows_num)
    columns_num = int(columns_num)
    cells = detect_split_into_cells(img, rows_num, columns_num)
    labels, label_image_map = cluster_cells(cells, clusters_num)

    for label in label_image_map.keys():
        img = label_image_map[label]
        img_no_bckg, bckg_color = detach_background(img)
        img_no_bckg = encode_base64(img_no_bckg)
        bckg_color = encode_base64(bckg_color)
        label_image_map[label] = (img_no_bckg, bckg_color)
    return labels, label_image_map


def encode_base64(input):
    input_base64 = base64.b64encode(input)
    input_base64 = input_base64.decode('ascii')
    # string_repr = base64.binascii.b2a_base64(image).decode("ascii")
    # encoded_img = np.frombuffer(base64.binascii.a2b_base64(string_repr.encode("ascii")))
    # image.save(byte_arr, format='PNG')  # convert the image to byte array
    # encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    return input_base64


def detect_split_into_cells(img, rows_num, columns_num):
    global imgWarpColored
    w, h, d = img.shape
    showImage(img)
    imgThresholdBinInvOtsu = preProcess(img)
    # showImage(imgThreshold)

    # Find contours
    imgContours = img.copy()
    imgBigContour = img.copy()
    # contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(imgThresholdBinInvOtsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    showImage(imgContours)
    imageList = ([img, imgThresholdBinInvOtsu, imgContours])

    # Biggest contour
    biggest, maxArea = findBiggestContour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 10)
        showImage(imgBigContour)
        rectangle_pts = get_rectangle_points(biggest)
        w = rectangle_pts[3, 0]
        h = rectangle_pts[3, 1]

        w = change_num(w, columns_num)
        h = change_num(h, rows_num)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (w, h))
        cv2.imwrite('imgWarpColored.png', imgWarpColored)

        showImage(imgWarpColored)

    # Splitting
    # img2 = img.copy()

    cells = split_into_cells(imgWarpColored, rows_num, columns_num)
    return cells


def cluster_cells(cells, cluster_count):
    cluster_count = int(cluster_count)
    label_to_symbol_map = {}

    # временный код для добавления помеченных изображений
    cells = upload_images_to_cells(cells, for_blank=True)
    result = clusterize_images(cells, cluster_count)
    labels_all = result["labels"]
    # последним у нас был помеченный пустой символ, находим его label
    label_of_blank = labels_all[cell.shape - 1]

    # удаляем последний label, так как он помеченный
    del label_of_blank[-1]
    label_to_symbol_map[label_of_blank] = blank_symbol_code

    # отделяем изображения, на которых есть символы (которые не blank)
    cells_without_blank = []
    for idx, label in enumerate(labels):
        if label != label_of_blank:
            cells_without_blank.append(cells[idx])

    # добавляем non blank помеченные изображения
    cells_without_blank = upload_images_to_cells(cells_without_blank, for_blank=False)
    result, cl = clusterize_images(cells_without_blank, cluster_count-1)
    labels_without_blank = result["labels"]
    label_elka = labels_without_blank[cells_without_blank.shape-5]
    label_flower = labels_without_blank[cells_without_blank.shape-4]
    label_kolos = labels_without_blank[cells_without_blank.shape-3]
    label_square = labels_without_blank[cells_without_blank.shape-2]
    label_sun = labels_without_blank[cells_without_blank.shape-1]

    # удаляем последние 5 элементов, так как они помеченные
    del labels_without_blank[-5:]
    label_to_symbol_map[label_elka] = elkaSymbolCOde
    label_to_symbol_map[label_flower] =
    label_to_symbol_map[label_kolos]
    label_to_symbol_map[label_square]
    label_to_symbol_map[label_sun]

    # TODO: из labels_all заменить все non-blank label-ы по очереди на label-ы из labels_without_blank

    # конец временного кода


    labels = cl.results_unique['labels']
    idxs = cl.results_unique['idx']
    label_image_map = {}
    for i, label in enumerate(labels):
        print(cells[i])
        label = int(label)
        label_image_map[label] = cells[idxs[i]]

    # return result['labels'], label_image_map
    for idx, cluster in enumerate(result["labels"]):
        dir_name = 'Clustered Images 60 epoch compose update/' + str(cluster)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        cv2.imwrite(dir_name + '/' + str(idx) + '.png', cells[idx])

    cl.scatter(zoom=4)
    cl.dendogram()
    cl.pca.plot()
    cl.pca.scatter(legend=False, label=False)
    cl.clusteval.plot()


# временный метод для добавления помеченных изображений к изображениям ячеек
# for_blank = True, если добавляем к cells изображение пустой ячейки (blank.jpg)
# for_blank = False, если добавляем изображения непустых ячеек
def upload_images_to_cells(cells, for_blank):
    folder = "chashka_symbols"
    images = []
    filenames = os.listdir(folder)

    if for_blank:
        filenames = ["blank.jpg"]
    else:
        filenames.remove("blank.jpg")

    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    cells = list(cells)
    cells.extend(images)
    return cells


def clusterize_images(images, cluster_count):
    sobel_images = []
    for img in images:
        img = preprocess.resize_and_RGB(img)
        img = preprocess.blur_and_Sobel(img)
        plt.show()
        img = img.flatten()
        sobel_images.append(img)

    sobel_images = np.array(sobel_images)
    cl = Clustimage(method='pca', params_pca={'n_components': cluster_count * 3}, dim=(128, 128))
    result = cl.fit_transform(sobel_images, cluster_count, cluster_count + 1)
    return result, cl


def detach_background(image):
    (h, w) = image.shape[:2]
    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=2)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # find most frequent color
    colors, counts = np.unique(quant, axis=0, return_counts=True)
    ind = np.argmax(counts)
    most_frequent_color = colors[ind]

    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    # delete background color
    mask_without_bckgrd = cv2.inRange(quant, most_frequent_color, most_frequent_color)
    # mask_without_bckgrd = cv2.cvtColor(mask_without_bckgrd, cv2.COLOR_GRAY2RGB)
    mask_without_bckgrd = mask_without_bckgrd - 255

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

    image_without_bckgrd = cv2.bitwise_and(image, image, mask=mask_without_bckgrd)

    return image_without_bckgrd, most_frequent_color
