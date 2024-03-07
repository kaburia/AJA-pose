from google.cloud import storage

def download_file_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

if __name__ == '__main__':
    bucket_name = "figures-gp/animal-kingdom/"
    source_blob_name = 'dataset.tar.gz'
    destination_file_name = 'dataset.tar.gz'

    download_file_from_gcs(bucket_name, source_blob_name, destination_file_name)
