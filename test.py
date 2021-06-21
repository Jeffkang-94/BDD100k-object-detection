from mmdet.apis import init_detector, inference_detector, show_result_pyplot
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_bdd.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
#checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_bdd_daynight/epoch_12.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
result = inference_detector(model, '/mnt2/datasets/VOC2012/sim10k2bdd/baseline_sim2bdd/original_sim10k/3398371.jpg')
show_result_pyplot(model, '/mnt2/datasets/VOC2012/sim10k2bdd/baseline_sim2bdd/original_sim10k/3398371.jpg', result, score_thr=0.5)