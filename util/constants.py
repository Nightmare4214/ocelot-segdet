from pathlib import Path

# Grand Challenge folders were input files can be found
GC_CELL_FPATH = Path("/home/icml007/Nightmare4214/datasets/ocelot2023_v1.0.1/images/test/cell")
GC_TISSUE_FPATH = Path("/home/icml007/Nightmare4214/datasets/ocelot2023_v1.0.1/images/test/tissue")

GC_METADATA_FPATH = Path("/home/icml007/Nightmare4214/datasets/ocelot2023_v1.0.1/metadata.json")

# Grand Challenge output file
GC_DETECTION_OUTPUT_PATH = Path("test/output/test_cell_classification.json")

# Sample dimensions
SAMPLE_SHAPE = (1024, 1024, 3)

# Tissue classes to predict
TISSUE_CLASSES = ['Background', 'Cancer', 'Other']
TISSUE_CLASS_COLOURS = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]
CELL_CLASSES = ['Background', 'Background_Cell', 'Tumour_Cell']
CELL_CLASS_COLOURS = [(0, 0, 0), (255, 255, 0), (0, 0, 255)]

# Dataloading/Model keys
SEG_MASK_LOGITS_KEY = 'seg_mask_logits'     # Segmentation logits
SEG_MASK_INT_KEY = 'seg_mask_int'           # Integer encoded segmentation mask
SEG_MASK_PROB_KEY = 'seg_mask_prob'         # Softmaxed segmentation mask
INPUT_IMAGE_KEY = 'input_image'             # RGB image
INPUT_MPP_KEY = 'input_mpp'                 # MPP of the input image (for dataloading)
INPUT_MASK_PROB_KEY = 'input_mask_prob'     # Softmaxed segmentation mask (at input)
INPUT_IMAGE_MASK_KEY = 'input_image_with_mask'  # Channel-wise concatenated RGB image and seg mask
POINT_HEATMAP_KEY = 'seg_mask_point_heatmap'    # Segmentation mask representing a point heatmap
DET_POINTS_KEY = 'det_points'       # Detected point coordinates (x, y)
DET_INDICES_KEY = 'det_indices'     # Class indices of detected points
DET_SCORES_KEY = 'det_scores'       # Confidence scores per-detection
GT_SEG_MASK_KEY = 'gt_seg_mask'     # Ground truth segmentation mask
GT_POINTS_KEY = 'gt_points'         # Ground truth point coordinates (x, y)
GT_INDICES_KEY = 'gt_indices'       # Ground truth class indices of ground truth points
GT_POINT_HEATMAP_KEY = 'gt_seg_mask_point_heatmap'  # Ground truth segmentation point heatmap
