from Arcface_feature import ArcFace
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
# from yolov5_face.Yolodetect import Yolov5
import faiss
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, max_wh - w -hp, max_wh - h - vp)
		return F.pad(image, padding, 255, 'constant')
if __name__ == '__main__':
	# detector = Yolov5('yolov5_face/weights/yolov5n-face.pt',640)
	extractor = ArcFace('r50','ms1mv3_arcface_r50_fp16/backbone.pth')
	faiss_index = faiss.read_index('db_face.index')
	with open('AFAD_face.txt','r') as f:
		db_images = [line.strip('\n') for line in f.readlines()]
	with open('log_detect.txt','r') as f:
		lines = [line.strip('\n') for line in f.readlines()]
	dict_frame_box = {}
	for line in lines:
		x,y,xmax,ymax, score, frame = line.split(' ')
		frame = int(frame)
		x = int(float(x))
		y = int(float(y))
		xmax = int(float(xmax))
		ymax = int(float(ymax))
		# if not frame in dict_frame_box:
		dict_frame_box[frame] = [x,y,xmax, ymax]
		# else:
			# dict_frame_box.append([x,y,xmax, ymax])
	vid_path = '20220605_164723_1.mp4'
	vid = cv2.VideoCapture(vid_path)
	num_frame = 0
	foucc = cv2.VideoWriter_fourcc('M','P','E','G')
	writer = cv2.VideoWriter('demo_recognize.mp4', foucc, 30, (720, 1280))
	while(True):
		ret, frame = vid.read()
		if not ret:
			break
		frame_copy = frame.copy()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		x,y,xmax, ymax = dict_frame_box[num_frame]
		frame = frame[y:ymax, x:xmax]

		feat = extractor.extract_feature([frame])
		score, idx = faiss_index.search(feat,1)
		score = score[0][0]
		idx = idx[0][0]
		frame_copy = cv2.rectangle(frame_copy, (x, y), (xmax, ymax), (0,255,0),5)
		# if score > 400:
		name_id = db_images[idx]
		name_id = name_id.split('/')[-2]
		if name_id == 'congtt':
			frame_copy = cv2.putText(frame_copy,name_id,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0, 255), 2, cv2.LINE_AA)
		with open('log_recogn.txt','a') as f:
			print(name_id, score,file=f)
		cv2.namedWindow('img', cv2.WINDOW_NORMAL)
		cv2.imshow('img', frame_copy)
		cv2.waitKey(5)
		num_frame +=1
		writer.write(frame_copy)
		
