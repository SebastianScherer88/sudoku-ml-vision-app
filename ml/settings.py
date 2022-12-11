import logging

# processing
LOG_LEVEL = logging.DEBUG

logger = logging.getLogger('processing')
logger.setLevel(LOG_LEVEL)

LOCAL_TEMP_DIR = './temporary'

# bucket
S3_DATA_BUCKET = 's3://bettmensch88-aws-dev-bucket'

# bucket dirs & files
S3_PROJECT_DIR = f'{S3_DATA_BUCKET}/sudoku-ml-vision'

# rar file
S3_CELL_DIGIT_CLASSIFICATION_DIR = f'{S3_PROJECT_DIR}/cell-digit-classification'
S3_CELL_DIGIT_CLASSIFICATION_SOURCE_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/source_rar'
S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE = '10000.rar'

# tf train & validate records dataset
IMAGE_RECORD_DATASETS_DIMENSION = (100,100)
IMAGE_CHANNEL_N = 3
IMAGE_RECORD_DATASETS_BATCH_SIZE = 32
S3_CELL_DIGIT_CLASSIFICATION_TF_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/train_validate_original_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN = 'train_original_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE = 'validate_original_tf'

# tf rotations train & validate tf records dataset
S3_CELL_DIGIT_CLASSIFICATION_ROTATED_TF_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/rotated_train_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_ROTATED = 'rotated_train_tf'

# tf blank images train & validate dataset
S3_CELL_DIGIT_CLASSIFICATION_BLANK_TF_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/blank_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_BLANK = 'blank_train_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_BLANK = 'blank_validate_tf'

# tf all images train & validate dataset
S3_CELL_DIGIT_CLASSIFICATION_BLANK_TF_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/all_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_ALL = 'train_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_ALL = 'validate_tf'


S3_IMAGE_SEGMENTATION_DIR = f'{S3_PROJECT_DIR}/image-segmentation'


RANDOM_SEED = 35

# modelling
MODEL_DIR = 'ml_models'