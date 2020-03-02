"""
    图像处理模块
"""
# coding=utf-8
import cv2
import numpy as np


def reshapeImg(img):
    """
        修改图像尺寸为512X384
    """
    height = img.shape[0]
    width = img.shape[1]
    if width > height:
        img = np.rot90(img, 1)
    img = cv2.resize(img, (384, 512),interpolation = cv2.INTER_AREA)
    return img


def MaxAreaContour(contours):
    """
        寻找最大面积轮廓
    """
    while len(contours) != 1:
        if cv2.contourArea(contours[1]) > cv2.contourArea(contours[0]):
            del contours[0]
        else:
            del contours[1]
    return contours


def getGrayDiff(image, currentPoint, tmpPoint):
    """
        区域生长计算差值
    """
    return abs(int(image[currentPoint[0], currentPoint[1]]) - int(image[tmpPoint[0], tmpPoint[1]]))


def regional_growth(gray, seeds, threshold=15) :
    """
        区域生长算法
    """
    # 每次区域生长的时候的种子像素之间的八个邻接点
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), \
                        (0, 1), (-1, 1), (-1, 0)]
    threshold = threshold # 种子生长时候的相似性阈值，默认即灰度级不相差超过15以内的都算为相同
    height, weight = gray.shape
    seedMark = np.zeros(gray.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)   # 将种子添加到种子的列表中
    label = 1	# 标记点的flag
    while(len(seedList)>0):     # 如果种子列表里还存在种子点
        currentPoint = seedList.pop(0)  # 将最前面的那个种子抛出
        seedMark[currentPoint[0], currentPoint[1]] = label   # 将对应位置的点标志为1
        for i in range(8):  # 对这个种子点周围的8个点一次进行相似性判断
            tmpX = currentPoint[0] + connects[i][0]
            tmpY = currentPoint[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:    # 如果超出限定的阈值范围
                continue    # 跳过并继续
            grayDiff = getGrayDiff(gray, currentPoint, (tmpX, tmpY))   # 计算此点与种子像素点的灰度级之差
            if grayDiff < threshold and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append((tmpX, tmpY))
    return seedMark


def Grad(gray):
    """
        计算图像梯度
    """
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)  # 对x求一阶导
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)  # 对y求一阶导
    gradx = cv2.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return gradxy


def Otsu(gray):
    """
        otsu阈值法去掉小于阈值部分，大于阈值部分保留原值
    """
    ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, gray = cv2.threshold(gray, ret, 255, cv2.THRESH_TOZERO)
    return gray


def DelGrid(data):
    """
        去掉数据图像中的格点
    """
    ret, bw = cv2.threshold(data, 2, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.erode(bw, kernel, iterations=1)
    image, contours, h = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 20:
            cv2.drawContours(mask, contours, i, 1, cv2.FILLED)
    mask = cv2.dilate(mask, kernel, iterations=1)
    data = data * mask
    return data


def ShowData(pic2):
    """
        去掉pic2中数据以外的无用数据与干扰
    """
    ret, bw = cv2.threshold(pic2, 1, 255, cv2.THRESH_BINARY)
    label_x = np.where(bw > 0)[1].max()
    bw[..., 70:] = 0
    for k in range(200):
        lines = cv2.HoughLines(bw, 1, np.pi / 180, k)
        if len(lines) == 1:
            break
    rho, theta = lines[0][0][0], lines[0][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * a)
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * a)

    mask = np.zeros(pic2.shape, dtype=np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
    non_zeros = np.where(mask != 0)

    for i in range(len(non_zeros[0])):
        if non_zeros[0][i] == (mask.shape[0] - 1):
            down_x = non_zeros[1][i]
            break

    for i in range(len(non_zeros[0])):
        row_pic2 = pic2[non_zeros[0][i]:non_zeros[0][i] + 1, ...]
        row_mask = mask[non_zeros[0][i]:non_zeros[0][i] + 1, ...]

        delta = down_x - non_zeros[1][i]
        if delta > 0:
            row_pic2[0][row_pic2.shape[1] - delta:] = 0
        if delta < 0:
            row_pic2[0][0:abs(delta)] = 0
        row_pic2 = np.roll(row_pic2, delta, axis=1)
        pic2[non_zeros[0][i]:non_zeros[0][i] + 1, ...] = row_pic2

        row_mask = np.roll(row_mask, delta, axis=1)
        mask[non_zeros[0][i]:non_zeros[0][i] + 1, ...] = row_mask

    mask[0, down_x] = 0
    for i in range(1, len(non_zeros[0] - 2)):
        u = pic2[i - 1:i + 2, down_x - 4:down_x + 5]
        if len(np.where(u != 0)[0]) < 2:
            mask[i, down_x] = 0

    y_up, y_down = np.where(mask != 0)[0].min(), np.where(mask != 0)[0].max()

    pic2[..., down_x - 5:down_x + 1] = 0

    data = pic2[..., down_x - 5:]
    data = DelGrid(data)
    pic2[..., down_x - 5:] = data

    g = int(pic2.max())
    cv2.line(pic2, (down_x, y_up), (down_x, y_down), g, 1)
    cv2.line(pic2, (down_x, y_down), (label_x, y_down), g, 1)

    return pic2


def CutScreen(img):
    """
        分割出屏幕部分
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, gray.max()/3, gray.max())

    center_row = int(np.where(edge > 0)[0].mean())
    center_col = int(np.where(edge > 0)[1].mean())

    kernel = np.ones((5, 5), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=1)
    edge[center_row - 32:center_row + 32, center_col - 32: center_col + 32] = 0

    seed_points = [(center_row, center_col)]
    bw = regional_growth(edge, seed_points, 100)
    bw = bw.astype(np.uint8)
    bw = bw * 255

    image, contours, h = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = MaxAreaContour(contours)

    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    mask = np.zeros(bw.shape, dtype=np.uint8)
    cv2.drawContours(mask, [box], 0, 3, cv2.FILLED)
    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (1, 1, img.shape[1], img.shape[0])
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img


def ProjectScreen(img, source):
    """
        通过投影变换校正屏幕
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    image, contours, h = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = MaxAreaContour(contours)

    for e in range(50):
        approx = cv2.approxPolyDP(contours[0], e, True)
        if len(approx) == 4:
            break
    corner = []
    for i in range(len(approx)):
        corner.append(approx[i][0].tolist())
    corner.sort(key=lambda x: x[1])
    if corner[0][0] > corner[1][0]:
        t = corner[0]
        corner[0] = corner[1]
        corner[1] = t
    if corner[2][0] > corner[3][0]:
        t = corner[2]
        corner[2] = corner[3]
        corner[3] = t

    w = img.shape[1]
    h = img.shape[0]
    presult = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    psource = np.float32(corner)
    A1 = cv2.getPerspectiveTransform(psource, presult)
    result = cv2.warpPerspective(source, A1, (w, h))
    return result


def RemoveBound(img):
    """
        去掉屏幕的边框
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    gradxy = Grad(gray)

    ret, bw = cv2.threshold(gradxy, 10, 255, cv2.THRESH_BINARY)
    bw = cv2.bitwise_not(bw)
    image, contours, h = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = MaxAreaContour(contours)

    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    corner = []
    for i in range(len(box)):
        corner.append(box[i].tolist())
    corner.sort(key=lambda x: x[1])
    if corner[0][0] > corner[1][0]:
        t = corner[0]
        corner[0] = corner[1]
        corner[1] = t
    if corner[2][0] > corner[3][0]:
        t = corner[2]
        corner[2] = corner[3]
        corner[3] = t

    w = img.shape[1]
    h = img.shape[0]
    presult = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    psource = np.float32(corner)
    pro_mat = cv2.getPerspectiveTransform(psource, presult)
    result = cv2.warpPerspective(img, pro_mat, (w, h))
    return result


def OutData(img):
    """
        输出最后数据
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if img[i][j][0] > img[i][j][2]:
                mk = img[i][j][0] - img[i][j][2]
                if gray[i][j] > mk:
                    gray[i][j] = gray[i][j] - mk
                else:
                    gray[i][j] = 0

    edge = cv2.Canny(gray, gray.max()/6, gray.max()/2)
    edge[0:20, ...] = 0
    edge[edge.shape[0] - 10:, ...] = 0
    edge[..., 0:10] = 0
    edge[0:60, edge.shape[1] - 60:] = 0

    kernel = np.ones((5, 5), np.uint8)
    bw = cv2.dilate(edge, kernel, iterations = 1)

    image, contours, h = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    while(i<len(contours)):
        if cv2.contourArea(contours[i]) < 100:
            del contours[i]
        else:
            i = i + 1
    mask = np.zeros(bw.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)

    gray_f = gray * mask
    gray_f = cv2.equalizeHist(gray_f)
    gray_f = gray_f.astype(np.float32)
    gray_f = gray_f / 255

    gray = gray.astype(np.float32)
    gray = gray / 255

    gray = 0.7 * gray + 0.3 * gray_f

    gray = gray * 255
    gray = gray.astype(np.uint8)

    cut_x1 = 90
    cut_x2 = 365
    cut_x3 = 415
    pic1 = gray[0:cut_x1, ...]
    pic1 = Otsu(pic1)
    pic1 = cv2.equalizeHist(pic1)
    gray[0:cut_x1, ...] = pic1

    pic3 = gray[cut_x2:cut_x3, ...]
    pic3 = Otsu(pic3)
    gray[cut_x2:cut_x3, ...] = pic3

    pic4 = gray[cut_x3:, ...]
    pic4 = Otsu(pic4)
    pic4[..., 0:120] = 0
    pic4[..., 260:] = 0
    pic4 = cv2.equalizeHist(pic4)
    gray[cut_x3:, ...] = pic4

    pic2 = gray[cut_x1:cut_x2, ...]
    pic2[..., 0:65] = Otsu(pic2[..., 0:65])
    pic2[..., 65:] = Otsu(pic2[..., 65:])
    pic2[..., 0:20] = 0
    pic2[..., pic2.shape[1] - 3:] = 0
    pic2 = ShowData(pic2)
    gray[cut_x1:cut_x2, ...] = pic2

    gray = cv2.bitwise_not(gray)
    return gray


def Process_24CGM(img):
    """
        24CGM处理方法
    """
    img_cut = CutScreen(img)
    img = ProjectScreen(img_cut, img)
    img = RemoveBound(img)
    img = OutData(img)
    return img