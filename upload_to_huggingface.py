from huggingface_hub import HfApi, upload_file

# Replace this with your Hugging Face repo name (make sure you've created it at huggingface.co/new)
repo_id = "ranitmaity/fashion-recommendation-system"

# Upload all 3 files
files_to_upload = ["model.h5", "embeddings.pkl", "filenames.pkl"]

for file in files_to_upload:
    upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=repo_id,
        repo_type="model"
    )

print("✅ All files uploaded to Hugging Face!")
