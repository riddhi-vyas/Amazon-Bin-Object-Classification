import boto3
import botocore
from botocore.handlers import disable_signing
import pandas as pd

resource = boto3.resource("s3")
resource.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)
bucket = resource.Bucket("aft-vbi-pds")

data = pd.read_csv("clean_dataset.csv")
file_list = data["filename"].tolist()
path = "bin-images/"
local_path = "bin_images/"
for item in file_list:
    print(f"File {item} downloaded!")

    try:
        bucket.download_file(f"{path}{item}", f"{local_path}{item}")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise
