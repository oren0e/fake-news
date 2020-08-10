import os
import boto3

S3_BUCKET = 'fake-news-assets'
AWS_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET = os.environ.get('AWS_SECRET_ACCESS_KEY')

session = boto3.Session(
    aws_access_key_id=AWS_ID,
    aws_secret_access_key=AWS_SECRET,
)