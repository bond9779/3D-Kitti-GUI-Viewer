import numpy as np 
import math 
import cv2 
import os
import matplotlib.pyplot as plt
import pdb

class Creator:
	def __init__(self):
		''' Initialize the necessary parameters for the visulaization '''
		self.BEV_HEIGHT = 1024
		self.BEV_WIDTH = 1024
		self.BEV_CHANNELS = 3
		self.GRID_HEIGHT = self.BEV_HEIGHT // 32
		self.GRID_WIDTH = self.BEV_WIDTH // 32

		self.BEV_Y_IDX_MAX = self.BEV_HEIGHT - 1
		self.BEV_X_IDX_MAX = self.BEV_WIDTH - 1
		self.RED_CHANNEL_IDX = 0
		self.GREEN_CHANNEL_IDX = 1
		self.BLUE_CHANNEL_IDX = 2

		self.LIDAR_X_IDX = 0
		self.LIDAR_Y_IDX = 1
		self.LIDAR_Z_IDX = 2
		self.LIDAR_R_IDX = 3

		self.bc = {}
		self.bc['minX'] = 0
		self.bc['maxX'] = 80 
		self.bc['X_len'] = self.bc['maxX'] - self.bc['minX']
		self.bc['X_off'] = -self.bc['minX']

		self.bc['minY'] = -40
		self.bc['maxY'] = 40
		self.bc['Y_len'] = self.bc['maxY'] - self.bc['minY']
		self.bc['Y_off'] = -self.bc['minY']

		self.bc['minZ'] = -3
		self.bc['maxZ'] = 1
		self.bc['Z_len'] = self.bc['maxZ'] - self.bc['minZ']
		self.bc['Z_off'] = -self.bc['minZ']

	def cropnormpc(self, pc):
		'''Crop the point cloud and normalize all dimensions to [0, 1].'''
		# remove the point out of range x,y,z
		mask = np.where((pc[:, self.LIDAR_X_IDX] >= self.bc['minX']) & (pc[:, self.LIDAR_X_IDX] <= self.bc['maxX']) & 
		                (pc[:, self.LIDAR_Y_IDX] >= self.bc['minY']) & (pc[:, self.LIDAR_Y_IDX] <= self.bc['maxY']) & 
		                (pc[:, self.LIDAR_Z_IDX] >= self.bc['minZ']) & (pc[:, self.LIDAR_Z_IDX] <= self.bc['maxZ']))
		npc = pc[mask]

		# shift origin to (0,0,0) and normalize the points to range [0,1]
		npc[:,0] = (npc[:,0] + self.bc['X_off'])/float(self.bc['X_len'])
		npc[:,1] = (npc[:,1] + self.bc['Y_off'])/float(self.bc['Y_len'])
		npc[:,2] = (npc[:,2] + self.bc['Z_off'])/float(self.bc['Z_len'])

		return npc

	def pctoBEV(self, npc):

		bev_img = np.zeros((self.BEV_HEIGHT, self.BEV_WIDTH, self.BEV_CHANNELS))
		PointCloud = npc
		PointCloud[:,0] = np.int_(np.round(PointCloud[:,0] * self.BEV_Y_IDX_MAX))
		PointCloud[:,1] = np.int_(np.round(PointCloud[:,1] * self.BEV_X_IDX_MAX))

		indices = np.lexsort((-PointCloud[:,2],PointCloud[:,1],PointCloud[:,0]))
		PointCloud = PointCloud[indices]

		unique, indices, counts = np.unique(PointCloud[:,0:2], axis = 0, return_index=True, return_counts=True)
		PointCloud_top = PointCloud[indices]

		normalizedCounts = np.minimum(1.0, np.log(counts + 1)/np.log(64))
		bev_img[np.int_(PointCloud_top[:,0]), 
		       np.int_(PointCloud_top[:,1]), 
		       self.RED_CHANNEL_IDX] = normalizedCounts

		bev_img[np.int_(PointCloud_top[:,0]), 
		       np.int_(PointCloud_top[:,1]), 
		       self.GREEN_CHANNEL_IDX] = PointCloud_top[:,2]

		indices = np.lexsort((-PointCloud[:,3],PointCloud[:,1],PointCloud[:,0]))
		PointCloud = PointCloud[indices]

		unique, indices, counts = np.unique(PointCloud[:,0:2], axis = 0, return_index=True, return_counts=True)
		PointCloud_top = PointCloud[indices]
		bev_img[np.int_(PointCloud_top[:,0]), 
		       np.int_(PointCloud_top[:,1]), 
		       self.BLUE_CHANNEL_IDX] = PointCloud_top[:,3]

		return bev_img

	def ry_to_rz(self, ry):
		angle = -ry - np.pi / 2

		if angle >= np.pi:
		    angle -= np.pi
		if angle < -np.pi:
		    angle = 2*np.pi + angle

		return angle

	def get_corners(self, x, y, w, l, yaw):
		bev_corners = np.zeros((4, 2), dtype=np.float32)

		# front left
		bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
		bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

		# rear left
		bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
		bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

		# rear right
		bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
		bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

		# front right
		bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
		bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

		return bev_corners


	def draw_BEV(self, labels, img):    


		for label in labels:
			obj_type, truncated, occluded, alpha, \
			cam_bbox_left, cam_bbox_top, cam_bbox_right, cam_bbox_bottom, \
			lidar_height, lidar_width, lidar_length, cam_x, cam_y, cam_z, cam_y_rotation = label[:-1].strip().split(' ')[:15]

			bb = []


			alpha = float(alpha)
			lidar_width = float(lidar_width)
			lidar_length = float(lidar_length)
			cam_x = float(cam_x)
			cam_y = float(cam_y)
			cam_z = float(cam_z)
			rz = float(cam_y_rotation) + 1.57

			bev_x = (-cam_x + self.bc['Y_off'])/float(self.bc['Y_len'])
			if bev_x < 0 or bev_x > 1:
			    continue
			bev_y = (cam_z + self.bc['X_off'])/float(self.bc['X_len'])
			if bev_y < 0 or bev_y > 1:
			    continue
			bev_x = bev_x * 32
			bev_y = bev_y * 16
			bev_w = lidar_width/float(self.bc['Y_len'])
			bev_h = lidar_length/float(self.bc['X_len'])
			# rz = np.pi * 2 - cam_y_rotation
			im = np.sin(rz)
			re = np.cos(rz)
			# theta = math.atan2(im,re)
			theta = np.arctan2(im, re)

			bev_x = int(np.round(bev_x * 32, 0))
			bev_y = int(np.round(bev_y * 64, 0))
			bev_w = int(np.round(bev_w * self.BEV_X_IDX_MAX, 0))
			bev_h = int(np.round(bev_h * self.BEV_Y_IDX_MAX, 0))

			corners = self.get_corners(bev_x, bev_y, bev_w, bev_h, theta).reshape(-1, 1, 2).astype(int)
			img = cv2.putText(img, obj_type, tuple(np.min(corners[:,0], axis = 0)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)
			# pdb.set_trace()

			cv2.polylines(img, [corners], True, (0,0,255), 1)
			    

		return img

	def bev_gen(self, pc, labels):
		pc = self.cropnormpc(pc)
		img = self.pctoBEV(pc)
		img = self.draw_BEV(labels, img)
		return img


	def roty(self, t):
		''' Rotation about the y-axis. '''
		c = np.cos(t)
		s = np.sin(t)
		return np.array([[c,  0,  s],
		                 [0,  1,  0],
		                 [-s, 0,  c]])

	def project_to_image(self, pts_3d, P):
		''' Project 3d points to image plane.

		Usage: pts_2d = projectToImage(pts_3d, P)
		  input: pts_3d: nx3 matrix
		         P:      3x4 projection matrix
		  output: pts_2d: nx2 matrix

		  P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
		  => normalize projected_pts_2d(2xn)

		  <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
		      => normalize projected_pts_2d(nx2)
		'''
		n = pts_3d.shape[0]
		pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
		#print(('pts_3d_extend shape: ', pts_3d_extend.shape))
		pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
		pts_2d[:,0] /= pts_2d[:,2]
		pts_2d[:,1] /= pts_2d[:,2]
		return pts_2d[:,0:2]

	def draw_projected_box3d(self, image, qs, color=(255,0,255), thickness=2):
		''' Draw 3d bounding box in image
		qs: (8,3) array of vertices for the 3d box in following order:
		    1 -------- 0
		   /|         /|
		  2 -------- 3 .
		  | |        | |
		  . 5 -------- 4
		  |/         |/
		  6 -------- 7
		'''
		try:
			qs = qs.astype(np.int32)
			for k in range(0,4):
			# Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
				i,j=k,(k+1)%4
				# use LINE_AA for opencv3
				cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

				i,j=k+4,(k+1)%4 + 4
				cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

				i,j=k,k+4
				cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
			return image
		except:
			return image


	def compute_box_3d(self, obj, P):
		''' Takes an object and a projection matrix (P) and projects the 3d
		    bounding box into the image plane.
		    Returns:
		        corners_2d: (8,2) array in left image coord.
		        corners_3d: (8,3) array in in rect camera coord.
		'''
		# compute rotational matrix around yaw axis
		obj_type, truncated, occluded, alpha, \
			cam_bbox_left, cam_bbox_top, cam_bbox_right, cam_bbox_bottom, \
			lidar_height, lidar_width, lidar_length, cam_x, cam_y, cam_z, cam_y_rotation = obj[:-1].strip().split(' ')[:15]

		alpha = float(alpha)
		lidar_width = float(lidar_width)
		lidar_length = float(lidar_length)
		cam_x = float(cam_x)
		cam_y = float(cam_y)
		cam_z = float(cam_z)
		rz = float(cam_y_rotation) + 1.57

		bev_x = (-cam_x + self.bc['Y_off'])/float(self.bc['Y_len'])
		if bev_x < 0 or bev_x > 1:
		    return 0, 0, None, None
		bev_y = (cam_z + self.bc['X_off'])/float(self.bc['X_len'])
		if bev_y < 0 or bev_y > 1:
		    return 0, 0, None, None
		R = self.roty(float(cam_y_rotation))

		# 3d bounding box dimensions
		l = float(lidar_length)
		w = float(lidar_width)
		h = float(lidar_height)

		# 3d bounding box corners
		x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
		y_corners = [0,0,0,0,-h,-h,-h,-h]
		z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
		    
		# rotate and translate 3d bounding box
		corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
		#print corners_3d.shape
		corners_3d[0,:] = corners_3d[0,:] + float(cam_x)
		corners_3d[1,:] = corners_3d[1,:] + float(cam_y)
		corners_3d[2,:] = corners_3d[2,:] + float(cam_z)
		#print 'cornsers_3d: ', corners_3d 
		# only draw 3d bounding box for objs in front of the camera
		if np.any(corners_3d[2,:]<0.1):
			corners_2d = None
			return corners_2d, np.transpose(corners_3d), 1, obj_type

		# project the 3d bounding box into the image plane
		corners_2d = self.project_to_image(np.transpose(corners_3d), P)
		#print 'corners_2d: ', corners_2d
		return corners_2d, np.transpose(corners_3d), 1, obj_type

	def inverse_rigid_trans(self, Tr):
		''' Inverse a rigid body transform matrix (3x4 as [R|t])
		    [R'|-R't; 0|1]
		'''
		inv_Tr = np.zeros_like(Tr) # 3x4
		inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
		inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
		return inv_Tr


	def cart2hom(self, pts_3d):
		"""
		:param pts: (N, 3 or 2)
		:return pts_hom: (N, 4 or 3)
		"""
		pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)))
		return pts_hom

	def project_rect_to_ref(self, pts_3d_rect, R0):
		''' Input and Output are nx3 points '''
		return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))

	def project_ref_to_velo(self, pts_3d_ref, C2V):
		pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
		return np.dot(pts_3d_ref, np.transpose(C2V))

	def project_rect_to_velo(self, pts_3d_rect, R0, V2C):
		pts_3d_ref = self.project_rect_to_ref(pts_3d_rect, R0)
		return self.project_ref_to_velo(pts_3d_ref, self.inverse_rigid_trans(V2C))


	def draw_cam2d(self, img, calib, labels):

		''' Show image with 2D bounding boxes '''
		for obj in labels:
			if obj.split(' ')[0] == 'DontCare':
				continue
			box3d_pts_2d, box3d_pts_3d, val, cls_ = self.compute_box_3d(obj, calib['P2'])
			img = self.draw_projected_box3d(img, box3d_pts_2d)
		return img


	def bbox_3d(self, calib, labels):
		classes = []
		bboxes = []
		for obj in labels:
			if obj.split(' ')[0] == 'DontCare':
				continue
			box3d_pts_2d, box3d_pts_3d, val, cls_ = self.compute_box_3d(obj, calib['P2'])
			if val == 1:
				box3d_pts_3d_velo = self.project_rect_to_velo(box3d_pts_3d, calib['R_rect'], calib['Tr_velo2cam'])
				# pdb.set_trace()
				bboxes.append(box3d_pts_3d_velo)
				classes.append(cls_)
		return bboxes, classes



	def read_lidar(self, path):
		pc = np.fromfile(path, dtype = np.float32).reshape(-1,4)
		return pc[(pc[:,0] > self.bc['minX']) * (pc[:,0] < self.bc['maxX']) * (pc[:,1] > self.bc['minY']) * (pc[:,1] < self.bc['maxY']) * (pc[:,2] > self.bc['minZ']) * (pc[:,2] < self.bc['maxZ'])]


	def read_label(self, path):
		with open(path,'r') as f:
			labels = f.readlines()
		return labels

	def read_calib(self, path):
		with open(path) as f:
			lines = f.readlines()

		obj = lines[2].strip().split(' ')[1:]
		P2 = np.array(obj, dtype=np.float32)
		obj = lines[3].strip().split(' ')[1:]
		P3 = np.array(obj, dtype=np.float32)
		obj = lines[4].strip().split(' ')[1:]
		R0 = np.array(obj, dtype=np.float32)
		obj = lines[5].strip().split(' ')[1:]
		Tr_velo_to_cam = np.array(obj, dtype=np.float32)

		return {'P2': P2.reshape(3, 4),
		        'P3': P3.reshape(3, 4),
		        'R_rect': R0.reshape(3, 3),
		        'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


	def read_img(self, path):
		return cv2.imread(path)



                




class Generator(Creator):
	def __init__(self, folder):
		super(Generator, self).__init__()
		self.main_folder = folder
		self.velodyne_folder = os.path.join(self.main_folder, 'velodyne')
		self.image_folder = os.path.join(self.main_folder, 'image_2')
		self.label_folder = os.path.join(self.main_folder, 'label_2')
		self.calib_folder = os.path.join(self.main_folder, 'calib')
		with open(os.path.join(self.main_folder, 'list.txt'), 'r') as f:
			self.list_of_files = f.readlines()
		self.index = 0
		self.pc = self.read_lidar(os.path.join(self.velodyne_folder, self.list_of_files[self.index][:-1] + '.bin'))
		self.label = self.read_label(os.path.join(self.label_folder, self.list_of_files[self.index][:-1] + '.txt'))
		self.calib = self.read_calib(os.path.join(self.calib_folder, self.list_of_files[self.index][:-1] + '.txt'))
		self.image = self.read_img(os.path.join(self.image_folder, self.list_of_files[self.index][:-1] + '.png'))
		self.filename = self.list_of_files[self.index]
		self.bev_img = self.bev_gen(self.pc, self.label)
		self.cam_img = self.draw_cam2d(self.image, self.calib, self.label)
		plt.imsave('temp.png', cv2.resize(self.bev_img, (640,480)))
		plt.imsave('temp1.png', cv2.resize(self.cam_img, (640,480)))
		self.bbox, self.classes = self.bbox_3d(self.calib, self.label)

	def getNextImage(self):
		if self.index < len(self.list_of_files):
			self.index += 1
			self.pc = self.read_lidar(os.path.join(self.velodyne_folder, self.list_of_files[self.index][:-1] + '.bin'))
			self.label = self.read_label(os.path.join(self.label_folder, self.list_of_files[self.index][:-1] + '.txt'))
			self.calib = self.read_calib(os.path.join(self.calib_folder, self.list_of_files[self.index][:-1] + '.txt'))
			self.image = self.read_img(os.path.join(self.image_folder, self.list_of_files[self.index][:-1] + '.png'))

			self.bev_img = self.bev_gen(self.pc, self.label)
			self.cam_img = self.draw_cam2d(self.image, self.calib, self.label)
			plt.imsave('temp.png', cv2.resize(self.bev_img, (640,480)))
			plt.imsave('temp1.png', cv2.resize(self.cam_img, (640,480)))
			self.bbox, self.classes = self.bbox_3d(self.calib, self.label)
			
			return 1
		return 0, 0

	def getPreviousImage(self):
		if self.index == 0:
			return 0, 0
		self.index -= 1
		self.pc = self.read_lidar(os.path.join(self.velodyne_folder, self.list_of_files[self.index][:-1] + '.bin'))
		self.label = self.read_label(os.path.join(self.label_folder, self.list_of_files[self.index][:-1] + '.txt'))
		self.calib = self.read_calib(os.path.join(self.calib_folder, self.list_of_files[self.index][:-1] + '.txt'))
		self.image = self.read_img(os.path.join(self.image_folder, self.list_of_files[self.index][:-1] + '.png'))

		self.bev_img = self.bev_gen(self.pc, self.label)
		self.cam_img = self.draw_cam2d(self.image, self.calib, self.label)

		plt.imsave('temp.png', cv2.resize(self.bev_img, (640,480)))
		plt.imsave('temp1.png', cv2.resize(self.cam_img, (640,480)))
		self.bbox, self.classes = self.bbox_3d(self.calib, self.label)
		return 1