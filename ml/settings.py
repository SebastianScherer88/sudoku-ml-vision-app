from logging import DEBUG, INFO, WARNING

# processing
LOG_LEVEL = DEBUG

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
IMAGE_RECORD_DATASETS_BATCH_SIZE = 32
S3_CELL_DIGIT_CLASSIFICATION_TF_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/train_validate_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN = 'train_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE = 'validate_tf'

# tf blank images train & validate dataset
S3_CELL_DIGIT_CLASSIFICATION_BLANK_TF_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/blank_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_BLANK = 'blank_train_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_VALIDATE_BLANK = 'blank_train_tf'

# tf rotations train & validate tf records dataset
S3_CELL_DIGIT_CLASSIFICATION_ROTATED_TF_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/rotated_tf'
S3_CELL_DIGIT_CLASSIFICATION_TF_TRAIN_ROTATED = 'rotated_train_tf'


S3_IMAGE_SEGMENTATION_DIR = f'{S3_PROJECT_DIR}/image-segmentation'

IMAGE_SOURCE_DATA_DIR = 'C:/Users/Sebastian.Scherer/Projects/sudoku_solver/data/digit_classification/source/250000_Final'
IMAGE_SYNTHETIC_DATA_DIR = 'C:/Users/Sebastian.Scherer/Projects/sudoku_solver/data/digit_classification/synthetic'

RANDOM_SEED = 35

BLANK_IMAGE_DIR = 'blank'
BLANK_DIGIT = 10
N_BLANK_IMAGES = 2000

ROTATED_IMAGE_DIR = 'rotated'
N_ROTATED_IMAGES_PER_DIGIT = 2000
ANGLE_RANGE = (10, 45)

# modelling