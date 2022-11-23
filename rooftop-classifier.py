import detectron2

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
import glob
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.data.catalog import Metadata


cfg = get_cfg()   

cfg.merge_from_file("config.yaml")   
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set the testing threshold for this model
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = 'model_final.pth'

predictor = DefaultPredictor(cfg) 

meta = Metadata(evaluator_type='coco',thing_classes=['Building-Roof', 'Building-Roof', 'Commercial-Flat-Roof', 'Commercial-Slope-Roof', 'Construction-Area', 'Flat Roof',
  'Playground', 'Slope-Flat-Roof', 'Slope-Roof', 'Solar-Flat-Roof', 'Solar-Pannel-Ground', 'Solar-Slope-Roof', 'TreeShading-Slope-Roof', 'Unknownshape-Roof'], 
  thing_dataset_id_to_contiguous_id={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13})
  

idx =1 
for imageName in glob.glob(f'./mapbox_img_*jpeg'):
  im = cv2.imread(imageName)
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1], metadata = meta)
  print(outputs)
  pixel_area = outputs["instances"].to("cpu").pred_masks.size()[0]
 
  if pixel_area == 0 :
    print('No Segmentation/Classification')
  else:
    #print(f'Segmented Area : {(outputs["instances"].to("cpu").pred_masks[0].numpy().sum())*.281} m') 
    print(f'Segmented Area : {(np.sqrt(outputs["instances"].to("cpu").pred_masks[0].numpy().sum()* .281)):.2f} m')
  
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  print("output_mapbox"+str(idx)+".jpeg")
  cv2.imwrite("output_mapbox"+str(idx)+".jpeg", out.get_image()[:, :, ::-1])
  idx =idx +1
  #cv2.imshow(out.get_image()[:, :, ::-1])