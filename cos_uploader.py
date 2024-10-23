from qcloud_cos import CosConfig, CosS3Client
from qcloud_cos.cos_exception import CosClientError, CosServiceError
import sys
import os
import logging
import time
import random
import yaml

class COSUploader:
    def __init__(self, config_path='config.yaml'):
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        secret_id = config['cos']['secret_id']
        secret_key = config['cos']['secret_key']
        region = config['cos']['region']
        token = None
        scheme = 'https'

        cos_config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)

        self.cos_setting = {
            'bucket': config['cos']['bucket']
        }

        os.environ['https_proxy'] = ''
        os.environ['http_proxy'] = ''
        self.client = CosS3Client(cos_config)

    def upload_file(self, dir_path: str, file_name: str) -> str:
        timestamp = int(time.time())
        object_key = f"{time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(timestamp))}-{random.randint(100000, 999999)}"

        try:
            response = self.client.upload_file(
                Bucket=self.cos_setting['bucket'],
                Key=dir_path + file_name,
                LocalFilePath=file_name,
                EnableMD5=False,
                progress_callback=None
            )
            print(f"成功上传 {object_key}:")
            print(response)
            return response['Location']
        except (CosClientError, CosServiceError) as e:
            print(f"上传 {object_key} 时发生错误:")
            print(e)
            return ""