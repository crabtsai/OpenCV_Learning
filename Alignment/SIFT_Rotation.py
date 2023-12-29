import cv2
import numpy as np
from matplotlib import pyplot as plt
#這是一個 FeatureMatching 類別的初始化方法，用來進行特徵匹配的準備工作。
class FeatureMatching:
    def __init__(self, query_image='data/query.jpg'):
        self.sift = cv2.SIFT_create()
        self.img_query = cv2.imread(query_image, 0)
        #讀取temp
        if self.img_query is None:
            print("Could not find train image " + query_image)
            raise SystemExit
        self.shape_query = self.img_query.shape[:2] # 對應的是y x
        # detectAndCompute 返回關鍵點，跟描述符，供後續特徵匹配做使用
        self.key_query, self.desc_query = self.sift.detectAndCompute(self.img_query, None)
        FLANN_INDEX_KDTREE = 0
        #FLANN 特徵匹配器
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) #使用K-D Tree算法進行最近鄰搜索，trees=5 表示構建K-D Tree時使用的樹的數量。
        search_params = dict(checks=50) #checks=50 表示進行搜索時，每次搜索最近鄰時檢查的節點數量。
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
 
    #抽取特徵
    def _extract_features(self, frame):
        sift = cv2.SIFT_create()
        key_train, desc_train = sift.detectAndCompute(frame, None)
        return key_train, desc_train
    #匹配特徵
    def _match_features(self, desc_frame):
        matches = self.flann.knnMatch(self.desc_query, desc_frame, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches
    def _detect_corner_points(self, key_frame, good_matches):
        src_points = np.float32([self.key_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) #是 query 圖像中好的匹配點對應的特徵點坐標
        dst_points = np.float32([key_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) #是 frame 圖像中好的匹配點對應的特徵點坐標
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0) #計算兩組對應點之間的單應變換，這裡用於計算 query 到 frame 的變換
        matchesMask = mask.ravel().tolist()
        h, w = self.img_query.shape[:2]
        src_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2) #是 query 圖像的四個角點的坐標，這裡是左上、左下、右下和右上。這些點用於形成一個矩形，後面會被變換到 frame 圖像上
        dst_corners = cv2.perspectiveTransform(src_corners, H) #得到在 frame 圖像上的角點坐標
        return dst_corners, H, matchesMask

    def _frontal_keypoints(self, frame, H):
        Hinv = np.linalg.inv(H) #計算單應性矩陣 H 的逆矩陣，得到透視變換的反變換矩陣。
        # 從單應性矩陣 H 中提取旋轉部分
        print(f'Hinv"{Hinv}')
        rotation_matrix = Hinv[:2, :2] # 從逆變換矩陣中提取旋轉部分，這是一個 2x2 的矩陣。
        # 計算旋轉角度（以度為單位）
        rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * (180 / np.pi) #使用反正切函數 arctan2 計算旋轉角度
        print(f"旋轉角度：{rotation_angle} 度")
        dst_size = frame.shape[:2]
        img_front = cv2.warpPerspective(frame, Hinv, dst_size, flags=cv2.INTER_LINEAR) #應用反變換，對輸入圖像進行逆透視變換，獲得正面視角的圖像。  
        return img_front , rotation_angle

    def match(self, frame):
        img_train = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape_train = img_train.shape[:2]
        key_train, desc_train = self._extract_features(img_train)
        good_matches = self._match_features(desc_train)
        if len(good_matches) < 4:
            self.num_frames_no_success += 1
            return False, frame
        # drawMatchesKnn 函數在兩張圖像之間繪製特徵匹配的結果, self.img_query：即模板圖像 img_train即待匹配的圖像
        img_match = cv2.drawMatchesKnn(self.img_query, self.key_query, img_train, key_train, [good_matches], None,
                                       flags=2)
        plt.imshow(img_match) #繪製匹配的結果
        plt.show()
        #算出變換後的角點座標，單映矩陣，matchesMask被用來只繪製模型的內點
        dst_corners, Hinv, _ = self._detect_corner_points(key_train, good_matches)
        dst_ravel = dst_corners.ravel()
        print(f'dst_ravel {dst_ravel}')
        #檢查角點座標是否超出範圍
        dst_ravel_Spec = 1500 #可允許範圍
        if (dst_ravel > shape_train[0] + dst_ravel_Spec).any() and (dst_ravel > -dst_ravel_Spec).any() \
                and (dst_ravel > shape_train[1] + dst_ravel_Spec).any():
            self.num_frames_no_success += 1
            print('角點離圖片太遠')
            return False, frame
        #轉換成正面視角
        img_front, img_angle = self._frontal_keypoints(frame, Hinv)
        plt.imshow(img_front)
        plt.show()
        output_path = "D:/CRAB-py/Alignment/123_1.jpg"
        cv2.imwrite(output_path, img_front)
        return True, img_front , img_angle
#TEST '20230920221112-000003_214567420070_Front_Suck1_Light0.bmp'
#載入原圖
img_train = cv2.imread('20230920221112-000003_214567420070_Front_Suck1_Light0.bmp')
#載入樣本圖
matching = FeatureMatching(query_image='160025.png')

flag , img_front, img_angle  = matching.match(img_train)

def rotate_image(image, angle): 
    # 取得影像中心點坐標
    center = tuple(np.array(image.shape[1::-1]) / 2)
    # 設定旋轉矩陣
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 進行影像旋轉
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    # 將旋轉後的影像存儲
    output_path = "D:/CRAB-py/Alignment/123.jpg"
    cv2.imwrite(output_path, rotated_image)
    return rotated_image

#算出的角度需要 換方向 加負號
rotate_image(img_train , -img_angle)
