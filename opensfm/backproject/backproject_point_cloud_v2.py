import numpy as np
import glob
from plyfile import PlyData, PlyElement
import cPickle as pickle
import sys
import os
import cv2
import json
#import pickle
from camera_types import denormalized_image_coordinates,get_shot_camera_data,load_equirectangular_camera,normalized_image_coordinates
from matplotlib import pyplot as plt


def filter_closest(source_world_points,source_whitened_points,source_camera_points, source_sphere_points, source_points_ids,source_colors,source_label_colors,source_mesh_ids,width,height):

        # idx works on np.array and not lists.
        camera_dist =np.reshape(np.linalg.norm(source_camera_points,axis=-1),(-1)) 
        dist_sorted_index = np.argsort(camera_dist)

	mesh_ids = source_mesh_ids[dist_sorted_index,:]        
        pts_ids = source_points_ids[dist_sorted_index]
        colors = source_colors[dist_sorted_index,:]
	colors_labels=source_label_colors[dist_sorted_index,:]
        world_pts = source_world_points[dist_sorted_index,:]
	whitened_pts=source_whitened_points[dist_sorted_index,:]
        pts_camera = source_camera_points[dist_sorted_index,:]
        pts_camera_sphere = source_sphere_points[dist_sorted_index,:]


        pts_ids_unique_info=np.unique(pts_ids, return_index=True)
        pts_ids_unique=pts_ids_unique_info[0]
        pts_ids_unique_index=pts_ids_unique_info[1]


        image_colors = np.zeros(shape=[height*width, 3],dtype=np.uint8)
	label_colors= np.zeros(shape=[height*width,3],dtype=np.uint8)
        points_3d_camera = np.zeros(shape=[height*width, 3],dtype=np.float32)
        points_3d_sphere =np.zeros(shape=[height*width, 3],dtype=np.float32)
        points_3d_world = np.zeros(shape=[height*width,3],dtype=np.float32)
	points_3d_world_whitened=np.zeros(shape=[height*width,3],dtype=np.float32)
	result_mesh_ids = np.zeros(shape=[height*width,1],dtype=np.int32)
        mask = np.zeros(shape=[height*width,1],dtype=np.uint8)

        image_colors[pts_ids_unique, :] = colors[pts_ids_unique_index,:]
	label_colors[pts_ids_unique,:] = colors_labels[pts_ids_unique_index,:]
        mask[pts_ids_unique,:]=255
        points_3d_camera[pts_ids_unique,:]=pts_camera[pts_ids_unique_index,:]
        points_3d_sphere[pts_ids_unique,:]= pts_camera_sphere[pts_ids_unique_index,:]
        points_3d_world[pts_ids_unique,:] =world_pts[pts_ids_unique_index,:]
	points_3d_world_whitened[pts_ids_unique,:]=whitened_pts[pts_ids_unique_index,:]
	result_mesh_ids[pts_ids_unique,:]=mesh_ids[pts_ids_unique_index,:]

        shp =(height,width,3)
        image_colors=np.reshape(image_colors,shp)
	label_colors=np.reshape(label_colors,shp)
        mask =np.reshape(mask,(height,width,1))
        points_3d_camera =np.reshape(points_3d_camera,shp)
        points_3d_sphere= np.reshape(points_3d_sphere,shp)
        points_3d_world=np.reshape(points_3d_world,shp)
	points_3d_world_whitened=np.reshape(points_3d_world_whitened,shp)
	result_mesh_ids = np.reshape(result_mesh_ids,(height,width,1))
	return points_3d_world,points_3d_world_whitened,points_3d_camera,points_3d_sphere,image_colors,label_colors,mask,result_mesh_ids
	# np.savez_compressed('../points_3d_info_new/'+short_file_name.split('.')[0]+'.npz',image_colors=image_colors,mask=mask,points_3d_camera=points_3d_camera,points_3d_sphere_unmasked=opensfm_sphere_coords,points_3d_sphere=points_3d_sphere,points_3d_world=points_3d_world,R=RT[:3,:3],T=RT[:3,3])
        #exit(1)

def get_masked_mesh_from_file(fname):
        mesh_fname='../segmentations_ply_strict/'+fname+'.npz'
        assert os.path.isfile(mesh_fname)
        result_mesh = np.load(mesh_fname)
	return result_mesh
def get_masked_mesh(fname,big_mesh):
	skip_colors=[(128, 64, 64),(128, 192, 192),(128, 64, 128),(192, 128, 0),(128, 0, 64),(0, 128, 128),(192, 0, 0),(128, 128, 192),(0, 0, 0), (0, 64, 64), (128, 128, 128)]
	color_info = pickle.load( open( "../segmentations/color_info.pic", "rb" ) )

	whitening_info=pickle.load( open( "../depthmaps/whitening_info.pic", "rb" ) )
	#print('whitenign insfo ' +str(whitening_info))
        assert os.path.isfile('../segmentations/'+fname), 'File not found: '+str(fname)
        label_image = cv2.imread('../segmentations/'+fname)
        #label_id_set = set((np.int32(label_image[:,:,2])*256*256+np.int32(label_image[:,:,1])*256+np.int32(label_image[:,:,0])).flatten().tolist())
	label_id_list=np.int32(label_image[:,:,0]).flatten().tolist()
        assert np.max(label_id_list)<256
	label_id_set=set(label_id_list)
	#print('Label id set: '+str(label_id_set))
	
	label_mask = big_mesh['labels'][:,0]==-1
	assert np.sum(label_mask)==0
	#print('label mask shape '+str(label_mask.shape))	
	for id in label_id_set:
		color = color_info['id_to_label'][id]['color']
		if(color in skip_colors):
			print('Skipping: '+str(color))		
			continue
		#print('id = '+str(id))
		temp_mask = big_mesh['labels'][:,0]==id
		#print('temp mask shape '+str(temp_mask)+ ' pas '+str(np.sum(temp_mask)))
		
		label_mask=label_mask | temp_mask
		#print('id = '+str(id)+'labels under mask : '+str(np.sum(label_mask)))

		color = whitening_info['id_to_color'][id]['color']		
		Kinv = whitening_info['id_to_color'][id]['K_inv']
		mean = whitening_info['id_to_color'][id]['mean']
		pts = big_mesh['points'][temp_mask,:]		
		pts_white  = big_mesh['points_whitened'][temp_mask,:]
		diff = np.sqrt(np.sum(np.square((np.matmul(Kinv,pts_white.T)+np.reshape(mean,(3,1))).T-pts),axis=-1))
		print('max diff' +str(np.max(diff)))
		assert np.max(diff)<0.000001
	#exit(1)
	
	result_mesh = {key:big_mesh[key][label_mask,:] for key in big_mesh.keys()}
	return result_mesh
def prepare_big_mesh_data(mesh_fname,color_fname):
	assert os.path.isfile(mesh_fname)
	assert os.path.isfile(color_fname)
	mesh_data = np.load(mesh_fname)
	result_data= {key:mesh_data[key] for key in mesh_data.keys()}
	add_label_colors(result_data,color_fname)
	result_data['mesh_ids'] =np.reshape( np.arange(mesh_data['points'].shape[0],dtype=np.int32),(mesh_data['points'].shape[0],1))
	return result_data
def add_label_colors(mesh_data,color_fname):
	#whitening_info=pickle.load( open( "../depthmaps/whitening_info.pic", "rb" ) )
	color_info = pickle.load( open( color_fname, "rb" ) )
	label_id_set = set(mesh_data['labels'].flatten().tolist())
	#print('ids available in mesh: '+str(label_id_set))
	mesh_data['label_colors'] = np.zeros(mesh_data['colors'].shape,dtype=np.uint8)
	#print('label colors shape '+str(mesh_data['label_colors'].shape))
	for id in label_id_set:
		mask = mesh_data['labels'][:,0]==id
		#print('mask shape '+str(mask.shape))
		mesh_data['label_colors'][mask,:]=color_info['id_to_label'][id]['color']
		#assert  whitening_info['id_to_color'][id]['color']==color_info['id_to_label'][id]['color']
	#exit(1)

def load_white_mean(whitening_fname):
	assert os.path.isfile(whitening_fname)
	whitening_info = pickle.load( open(whitening_fname, "rb" ) )
	print(whitening_info.keys())
	white_labels = []
	white_mean=[]
	white_Kinv=[]
	for color in sorted(whitening_info['color_to_id'].keys()):
	        white_labels.append(color)
	        white_mean.append(whitening_info['color_to_id'][color]['mean'])
	        white_Kinv.append(whitening_info['color_to_id'][color]['K_inv'])
	white_labels =np.array(white_labels,dtype=np.int32)
	white_mean =np.array(white_mean,dtype=np.float64)
	white_Kinv = np.array(white_Kinv,dtype=np.float64)
	return white_labels,white_mean,white_Kinv

print('args: '+str(sys.argv))
if(len(sys.argv)>2):
        modulus = int(sys.argv[1])
        remainder = int(sys.argv[2])
else:
        modulus=1
        remainder=0


#experiment_info={  'main_output_dir':'../points_3d_info_nosemanticfilter/'}

base_dir='/data/cvfs/ib255/shared_file_system/derivative_datasets/camvid_360_3d_data_final_sequences/camvid_360_final/TEST_STREETVIEW/'
EXPERIMENT_INFO={ 'part0102':{  'whitening_dir':'/data/cvfs/ib255/shared_file_system/derivative_datasets/camvid_360_3d_data_final_sequences/camvid_360_final/part_0102_seq_016E5_P1_R96_01500_04890/depthmaps/',
    		 	       'image_dir':base_dir+'part_0102_seq_016E5_P1_R96_01500_04890/'+'images_hand/'},
		}


main_output_dir='./test_01'

os.system('mkdir '+main_output_dir)
visual_output_dir=main_output_dir+'projections/'
os.system('mkdir '+visual_output_dir)

x_width=1920
x_height=960
pixel_coords= np.zeros((x_height,x_width,2))

print('Loading experiment info')
for e in EXPERIMENT_INFO.keys():
	einfo=EXPERIMENT_INFO[e]
	assert os.path.isdir(einfo['image_dir'])
	assert os.path.isdir(einfo['whitening_dir'])
	we,wm,wk=load_white_mean(einfo['whitening_dir']+'whitening_info.pic')
	einfo['white_labels']=we
	einfo['white_mean']=wm
	einfo['white_Kinv']=wk
	#big_mesh_data=prepare_big_mesh_data('../depthmaps/merged_whitened.npz')
	einfo['big_mesh_data']=prepare_big_mesh_data(einfo['whitening_dir']+'merged_whitened.npz',einfo['whitening_dir']+'../segmentations/color_info.pic')
print 'experiment info: '+str(EXPERIMENT_INFO['part05'].keys())
print('Filling coordinates')
for row in range(x_height):
	for col in range(x_width):
		pixel_coords [row,col,1]=row +0.5
		pixel_coords[row,col,0]=col +0.5
norm_pixel_coords= normalized_image_coordinates(np.reshape(pixel_coords,(x_width*x_height,2)),1920,960)

pr_count=-1
file_list=sorted(glob.glob('../experiments/*'))
total_count =len(file_list)
print('Starting to examine files')
for dirname in file_list:
	assert os.path.isdir(dirname)
	image_fname=dirname.split('/')[-1]+'.png'
	camera_data= np.load(dirname+'/feature_3d/camera_fit.npz')
        R,t=camera_data['R_opensfm'],camera_data['T_opensfm']
	shot_camera = load_equirectangular_camera({'width':x_width,'height':x_height})
	pr_count=pr_count+1
	print(str(pr_count)+ ' of '+str(total_count))
	if(pr_count % modulus != remainder):
		continue
	if(os.path.isfile(main_output_dir+image_fname.split('.')[0]+'.npz')) and (os.path.isfile(main_output_dir+'projections/'+image_fname)):
		print ('Skipping '+main_output_dir+image_fname.split('.')[0]+'.npz')
		continue
	print('File not found. Creating alignment: '+ main_output_dir+image_fname.split('.')[0]+'.npz')
	result_rgb = np.zeros((x_height,x_width,3))
	result_label = np.zeros((x_height,x_width,3))

	#mesh_data =get_masked_mesh_from_file(image_fname)	
	#mesh_data= get_masked_mesh(image_fname+'.png',big_mesh_data)

	einfo=None
	for eid in EXPERIMENT_INFO.keys():
		e=EXPERIMENT_INFO[eid]
		if os.path.isfile(e['image_dir']+image_fname):
			einfo=e
			break

	mesh_data=einfo['big_mesh_data']
	#exit(1)
	#mesh_data=big_mesh_data

	source_world_points = mesh_data ['points']
	source_image_colors = mesh_data['colors']
	source_label_colors = mesh_data['label_colors']
	source_whitened_points = mesh_data['points_whitened']
	source_meshids = mesh_data['mesh_ids']

	mul = np.matmul(R,np.reshape(source_world_points,(source_world_points.shape[0],3,1)))
	#print('mul '+str(mul.shape))
	source_camera_points= mul[:,:,0]+np.reshape(t,(1,3))
	dist_cameras = np.linalg.norm(source_camera_points,axis=-1)
	source_sphere_points =source_camera_points/np.reshape(dist_cameras,(source_camera_points.shape[0],1))
	cam_coords = shot_camera.project_many(source_camera_points)
	#print('cam coorsd shae '+str(cam_coords.shape))
        denom_cam_coords= denormalized_image_coordinates(cam_coords,shot_camera.width,shot_camera.height)
	#print('denom '+str(denom_cam_coords.shape))
	row = np.int32(denom_cam_coords[:,1])
	col = np.int32(denom_cam_coords[:,0])
	cid = np.arange(col.shape[0],dtype=np.int32)

	source_points_ids  = row * shot_camera.width+col

	points_3d_world,points_3d_whitened,points_3d_camera,points_3d_sphere,image_colors,label_colors,mask,mesh_ids= filter_closest(source_world_points,source_whitened_points,source_camera_points, source_sphere_points, source_points_ids,source_image_colors,source_label_colors,source_meshids,shot_camera.width,shot_camera.height)

	sphere_coords = np.reshape(shot_camera.pixel_bearing_many(norm_pixel_coords),(shot_camera.height,shot_camera.width,3))
	assert (np.min(np.abs(np.linalg.norm(sphere_coords,axis=-1)-1))<0.001)
	sphere_dist = np.sqrt(np.sum(np.square( sphere_coords-points_3d_sphere	),axis=-1))
	sphere_dist [mask[:,:,0]==0]=0
	#plt.imshow(sphere_dist)
	#plt.show()
	assert (np.max(sphere_dist)<0.004), 'Max dist '+str(np.max(sphere_dist))
        np.savez_compressed(main_output_dir+image_fname.split('.')[0]+'.npz',mesh_ids=mesh_ids,image_colors=image_colors,label_colors=label_colors,points_3d_world_whitened=points_3d_whitened,mask=mask,points_3d_camera=points_3d_camera,points_3d_sphere_unmasked=np.float32(sphere_coords),points_3d_sphere=points_3d_sphere,points_3d_world=points_3d_world,R=R,T=t,white_mean=einfo['white_mean'],white_Kinv = einfo['white_Kinv'],white_labels=einfo['white_labels'])

	#print('distance check is finished')
	result_rgb [row,col,:]=source_image_colors[cid,::-1]
	result_label[row,col,:]=source_label_colors[cid,:]
	cv2.imwrite(visual_output_dir+image_fname+'.png',result_rgb)
	cv2.imwrite(visual_output_dir+image_fname+'.seg.png',result_label)
	assert os.path.isfile(dirname+'/images/'+image_fname), 'File not found: '+dirname+'/images/'+image_fname
	os.system('cp '+dirname+'/images/'+image_fname+' ' +visual_output_dir+image_fname)
	#os.system('cp '+visual_label_dir+image_fname + ' '+visual_output_dir+image_fname+'.rseg.png')

	result_label=cv2.imread(visual_output_dir+image_fname+'.seg.png')
	#result_label[diff_img_mask&void_img_mask&void_img_mask_gt,:]=(0,0,255)
	#cv2.imwrite(visual_output_dir+image_fname+'.diff.png',result_label)
	#break
print('Finished')
