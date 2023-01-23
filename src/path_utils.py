import io
import json
import os
import boto3

def return_object_from_s3(path):
    with open(os.path.join("keys", "amazonkeys.json")) as config_file:
        key_dict = json.load(config_file)

    s3 = boto3.client('s3', aws_access_key_id=key_dict["aws_access_key_id"],
                      aws_secret_access_key=key_dict["aws_secret_access_key"])
    bucket_name = "rpmow"
    object = s3.get_object(Bucket=bucket_name, Key=path)
    return object['Body'].read().decode('utf-8')

if __name__ == '__main__':
    text = return_object_from_s3("hugo/5_complete_textref.txt")
    print(list(set(text)))
