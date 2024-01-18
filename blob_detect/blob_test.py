import cv2
import numpy as np

# 讀取圖像
image_path = "./download.jpg"  # 請替換為實際圖像的路徑
im = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 初始化 SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# 選擇檢測的標準（可以根據需要自行調整）
# 設定參數
params.filterByColor = True
params.blobColor = 0  # 若設定0 就是檢測所有顏色

params.minThreshold = 10 #最小，二值化閥值
params.maxThreshold = 200 #最大，二值化閥值

params.filterByArea = True
params.minArea = 10    # 最小面積
params.maxArea = 1000   # 最大面積

params.filterByCircularity = True
params.minCircularity = 0.5  # 最小圓形度

# 創建 SimpleBlobDetector 
detector = cv2.SimpleBlobDetector_create(params)

# 檢測關鍵點
keypoints = detector.detect(im)
# 提取 X 和 Y 座標
for point in keypoints:
    x, y = point.pt #(X,Y)座標
    x = round(x,2)
    y = round(y,2)
    Diameter = round(point.size,2) #直徑
    
    print(f"Keypoint at ({x}, {y}), Diameter: {Diameter}")

# 在圖像上繪製關鍵點
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 顯示原始圖像和帶有關鍵點的圖像
cv2.imshow("Original Image", im)
cv2.imshow("Image with Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
