import cv2
 
# 读取Dav格式视频
dav_file = 'test.dav'
cap = cv2.VideoCapture(dav_file)
 
# 获取视频的参数
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
# 设置输出视频的编解码器和参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'output.mp4'
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
 
# 逐帧读取视频并写入输出视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
 
# 释放资源
cap.release()
out.release()