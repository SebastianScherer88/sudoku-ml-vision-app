# processing

# bucket
S3_DATA_BUCKET = 's3://bettmensch88-aws-dev-bucket'

# bucket dirs & files
S3_PROJECT_DIR = f'{S3_DATA_BUCKET}/sudoku-ml-vision'

S3_CELL_DIGIT_CLASSIFICATION_DIR = f'{S3_PROJECT_DIR}/cell-digit-classification'
S3_CELL_DIGIT_CLASSIFICATION_SOURCE_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/source'
S3_CELL_DIGIT_CLASSIFICATION_SOURCE_FILE = '250000_Final.rar'
S3_CELL_DIGIT_CLASSIFICATION_EXTRACTED_SOURCE_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_SOURCE_DIR}/250000_Final'

S3_CELL_DIGIT_CLASSIFICATION_SYNTHETIC_DIR = f'{S3_CELL_DIGIT_CLASSIFICATION_DIR}/synthetic'

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