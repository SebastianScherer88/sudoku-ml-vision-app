from typing import Union
from pathlib import Path

import os

import boto3

from image_processing.settings import logger

def upload_directory(local_dir: Union[str,Path], s3_bucket: str, s3_dir: str):
    
    s3_client = boto3.client('s3')
    
    # upload all model related files
    for root,dirs,files in os.walk(local_dir):
        for file in files:
            
            local_file = os.path.join(root,file)
            
            logger.debug(f'Root: {root}. File: {file}')
            
            # remove local dir from root
            root_removed = root.replace(local_dir,'REPLACED_LOCAL_DIR').split('REPLACED_LOCAL_DIR')[-1]
            
            logger.debug(f'Root removed: {root_removed}')
            
            # convert os separators with s3 '/'
            root_replaced = root_removed.replace(os.sep,'/')
            
            logger.debug(f'Root replaced: {root_replaced}')
            
            # prepend with s3 'directory' and add file name 
            s3_file = f'{s3_dir}{root_replaced}/{file}'
            
            logger.debug(f'S3 file: {s3_file}')
            
            # upload
            s3_client.upload_file(local_file,s3_bucket,s3_file)

            logger.debug(f'Uploaded {local_file} to S3 at {s3_bucket}:{s3_file}')
            
def download_directory(s3_bucket: str, s3_dir: str, local_dir: Union[str,Path]):
    
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(s3_bucket)
    
    s3_dirname = s3_dir.split('/')[-1]
    
    logger.debug(f'S3 directory name: {s3_dirname}')
    
    for s3_obj in bucket.objects.filter(Prefix = s3_dir):
        
        s3_file = s3_obj.key
        
        logger.debug(f'S3 file: {s3_file}')
        
        s3_file_end = s3_file.split(f'{s3_dirname}/')[-1]
        
        logger.debug(f'S3 file suffix: {s3_file_end}.')
        
        local_file_end = s3_file_end.replace('/',os.sep)
        
        logger.debug(f'Local file relative path: {local_file_end}.')
        
        local_file = os.path.join(local_dir,s3_dirname,local_file_end)
        
        logger.debug(f'Local file path: {local_file}.')
        
        local_file_dir = os.path.dirname(local_file)
        
        logger.debug(f'Local file directory: {local_file_dir}.')
        
        if not os.path.exists(local_file_dir):
            os.makedirs(local_file_dir)
            
        bucket.download_file(s3_file, local_file) # save to same path
        
        logger.debug(f'Downloaded from S3 at {s3_bucket}:{s3_file} to {local_file}')