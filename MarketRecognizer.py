import cv2
import numpy as np
import math

ADAPTIVE_THRESH_SIZE = 35
APPROX_POLY_EPS = 0.08
MARKER_CELL_SIZE = 10
MARKER_SIZE = (7*MARKER_CELL_SIZE)


class Marker:
    m_id = 0
    m_corners = np.array([])
    # c0------c3
    # |		  |
    # |		  |
    # c1------c2

    def __init__(self):
        self.m_id = -1
        self.m_corners = np.array([
            [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0]
        ])

    def __init__(self, _id, _c0, _c1, _c2, _c3):
        self.m_id = _id
        self.m_corners = np.float32([
            _c0, _c1,
            _c2, _c3
        ])

    def estimateTransformToCamera(self, corners_3d, camera_matrix, dist_coeff, rmat, tvec):
        rot_vec=np.array([])
        # is ok?
        rmat=np.array([])
        cv2.solvePnP(corners_3d, self.m_corners, camera_matrix, dist_coeff, rot_vec, tvec)
        cv2.Rodrigues(rot_vec, rmat)
        return rmat

    def drawToImage(self, image, color, thickness):
        cv2.circle(image, tuple(self.m_corners[0]), thickness*2, color, thickness)
        cv2.circle(image, tuple(self.m_corners[1]), thickness, color, thickness)
        cv2.line(image, tuple(self.m_corners[0]), tuple(self.m_corners[1]), color, thickness, cv2.CV_AA)
        cv2.line(image, tuple(self.m_corners[1]), tuple(self.m_corners[2]), color, thickness, cv2.CV_AA)
        cv2.line(image, tuple(self.m_corners[2]), tuple(self.m_corners[3]), color, thickness, cv2.CV_AA)
        cv2.line(image, tuple(self.m_corners[3]), tuple(self.m_corners[0]), color, thickness, cv2.CV_AA)
        text_point = (np.array(self.m_corners[0]) + np.array(self.m_corners[2])) * 0.5
        ss = '%d' % self.m_id
        cv2.putText(image, ss, tuple(text_point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


class MarkerRecognizer:
    __m_marker_coords = np.array([])
    __m_markers = []

    # ret possible_markers
    def __markerDetect(self, img_gray, min_size, min_side_length):
        possible_markers = []
        # thresh_size = (min_size / 4) * 2 + 1
        _,img_bin = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        all_contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = []
        for i in range(0, len(all_contours)):
            if(len(all_contours[i]) > min_size):
                contours.append(all_contours[i])

        for i in range(0, len(contours)):
            eps = len(contours[i]) * APPROX_POLY_EPS
            approx_poly = cv2.approxPolyDP(contours[i], eps, True)

            if(len(approx_poly)!=4):
                continue
            if(not cv2.isContourConvex(approx_poly)):
                continue

            min_side = 3.4028e38
            for j in range(0, 4):
                side = approx_poly[j]- approx_poly[(j+1)%4]
                min_side = min(min_side, side[0].dot(side[0]))
            if min_side < min_side_length * min_side_length:
                continue

            marker = Marker(0, approx_poly[0][0], approx_poly[1][0], approx_poly[2][0], approx_poly[3][0])
            v1 = marker.m_corners[1] - marker.m_corners[0]
            v2 = marker.m_corners[2] - marker.m_corners[0]
            if np.cross(v1, v2) > 0:
                temp = marker.m_corners[3]
                marker.m_corners[3] = marker.m_corners[1]
                marker.m_corners[1] = temp
            possible_markers.append(marker)
        return possible_markers

    # ret final_markers
    def __markerRecognize(self, img_gray, possible_markers):
        final_markers = []

        bit_matrix = np.ndarray((5, 5), np.uint8)
        for i in range(0, len(possible_markers)):
            M = cv2.getPerspectiveTransform(possible_markers[i].m_corners, self.__m_marker_coords)
            marker_image = cv2.warpPerspective(img_gray, M, (MARKER_SIZE, MARKER_SIZE))
            _, marker_image = cv2.threshold(marker_image, 125, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

            flag = False
            for y in range(0, 7):
                inc = 6
                if(y==0 or y==6):
                    inc = 1
                cell_y = y * MARKER_CELL_SIZE

                for x in range(0, 7, inc):
                    cell_x = x * MARKER_CELL_SIZE
                    none_zero_count = cv2.countNonZero(marker_image[
                        cell_y:(cell_y + MARKER_CELL_SIZE), cell_x:(cell_x + MARKER_CELL_SIZE)
                    ])
                    if(none_zero_count > MARKER_CELL_SIZE * MARKER_CELL_SIZE / 4):
                        flag = True
                        break
                if(flag):
                    break
            if(flag):
                continue

            for y in range(0, 5):
                cell_y = (y + 1) * MARKER_CELL_SIZE

                for x in range(0, 5):
                    cell_x = (x + 1) * MARKER_CELL_SIZE
                    none_zero_count = cv2.countNonZero(marker_image[
                        cell_y:(cell_y + MARKER_CELL_SIZE), cell_x:(cell_x + MARKER_CELL_SIZE)
                    ])
                    if none_zero_count > MARKER_CELL_SIZE * MARKER_CELL_SIZE / 2:
                        bit_matrix[y, x] = 1
                    else:
                        bit_matrix[y, x] = 0

            good_marker = False
            for rotation_idx in range(0, 4):
                if self.__hammingDistance(bit_matrix) == 0:
                    good_marker = True
                    break
                bit_matrix = self.__bitMatrixRotate(bit_matrix)
            if not good_marker:
                continue

            final_marker = possible_markers[i]
            final_marker.m_id = self.__bitMatrixToId(bit_matrix)
            final_marker.m_corners = np.roll(final_marker.m_corners, -rotation_idx, axis=0)
            final_markers.append(final_marker)
        return final_markers

    def __markerRefine(self, img_gray, final_markers):
        for i in range(0, len(final_markers)):
            corners = final_markers[i].m_corners
            cv2.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    def __bitMatrixRotate(self, bit_matrix):
        out = bit_matrix.copy()
        rows, cols = bit_matrix.shape

        for i in range(0, rows):
            for j in range(0, cols):
                out[i, j] = bit_matrix[cols-j-1, i]
        return out

    def __hammingDistance(self, bit_matrix):
        ids = np.array([
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ])

        dist = 0
        for y in range(0, 5):
            minSum = 2147483640
            for p in range(0, 4):
                m_sum = 0
                for x in range(0, 5):
                    m_sum += not bit_matrix[y, x] == ids[p, x]
                minSum = min(minSum, m_sum)
            dist = dist + minSum
        return dist

    def __bitMatrixToId(self, bit_matrix):
        m_id = 0
        for y in range(0, 5):
            m_id <<= 1
            m_id = m_id | bit_matrix[y, 1]

            m_id <<= 1
            m_id = m_id | bit_matrix[y, 3]
        return m_id

    def __init__(self):
        self.__m_marker_coords = np.float32([
            [0, 0], [0, MARKER_SIZE-1],
            [MARKER_SIZE-1, MARKER_SIZE-1], [MARKER_SIZE-1, 0]
        ])

    def update(self, image_gray, min_size, min_side_length=10):
        img_gray = image_gray.copy()

        possible_markers = self.__markerDetect(img_gray, min_size, min_side_length)
        self.__m_markers = self.__markerRecognize(img_gray, possible_markers)
        self.__markerRefine(img_gray, self.__m_markers)

        return len(self.__m_markers)

    def getMarkers(self):
        return self.__m_markers

    def drawToImage(self, image, color, thickness):
        for i in range(0, len(self.__m_markers)):
            self.__m_markers[i].drawToImage(image, color, thickness)

    def getMarkersCenter(self):
        locs = {}
        for m_marker in self.__m_markers:
            dic = {}
            dic['pt'] = ( m_marker.m_corners[0] + m_marker.m_corners[1] + m_marker.m_corners[2] + m_marker.m_corners[3] ) / 4.0
            vec = m_marker.m_corners[3] - m_marker.m_corners[0]
            dic['angle'] = math.atan2(vec[1], vec[0]) / math.pi * 180
            locs[m_marker.m_id] = dic
        return locs
