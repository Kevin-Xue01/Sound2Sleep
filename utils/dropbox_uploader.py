import os
import dropbox
from cryptography.fernet import Fernet


ENCRYPTION_KEY = b'G3JvV1FItZ_PtdpMxDL1rO3mOJ4WbA2jzDoLmfvpsq0='
cipher = Fernet(ENCRYPTION_KEY)

#Hardcoded Dropbox Access Token
DROPBOX_ACCESS_TOKEN = "sl.u.AFmAMRSQg-oJa5bgjQmyoaC_hayKgNeGuyUi6TFCAfbOikQNQJmAcm9mkZPXY0TXuwllbJ-OHu8krY0DD-YhPmWHngj4uk2L2Xxw5Vs4BexiVNDc-dm58so-nJu89IEDVtY1iHk6JBlCRpZqHQYBWc7clNXwUng_GSfWcH8VbJsnmcuRdRCRfzDTM8wevvf8YzoTZg0We2K9WeCepHdBP0ybeyUb6Y-vQeAwIU_Rugl5exltzXUJjuj9YLZh95_zOWr0JNV3BnAfh9TDbX_tc3GrtV_Hfo6F8kksGxuflKwN922vsAWQZakIKDv32YH9ZI5HwKyUK-Y4JveHUYWSXJJ3q5GGNwlzJLUv_M2dUNvBZyeZPO2BrII0b_N-mnHZthScw-Nsm6DwRD2wGt-NbslW2Dhz6B9bWYBfBnMAoxOt0QqPqnt_RbnIK7PXr1GCdal65ybpxdJzkX66OGfOciKiB_RxsCznINGVz85Y3W1k_ho1v9z5KyKN_lyM3ZQnGIySVJbiRnuEy2N9kEouK0_SukNQar06gQ7AHB-0YUuXqR2_aRvMNUXwQ2cM-Pz2F2_nlt1oPslNhdyR7gD6DFcIhdEHKlKE8hPqm0uXKb-VwbfHK5qSl2Ksj8kkTcuA72k-PbCJVx7EHPLd9MDyg-p4EWkrs9YiUgz4ll68LoFcZ2YPActxZA3wvc5amLnjRNfVKKLr0ca5JHBbUCjOmPbaqPXYKDOh4opHaqoMNeHa42SQ6lv3BN1dHD5BqotW6qNZZ5_U8MKUoE1Q3xR-f5Po2f11BQwpRrZRXtdL2rw7rSa_WUN0A4XPojO9rf4f1z7iOblN02pLOZXCnnhNgYc2zZloRZxm3ML7QOwzBBfyGzVG5bfU-DdqMUgKdWvL2iFcYDTW6VnA-dmzW9M75PDNZbAD90G5si4VEmuWsoRNwYq5kt0zAgD4KD7eGYdAuOSvS3CMDne0I_rDGEELF2uCfVyNVaRvRX-dtKQIVoCXg2c_mg0xaun2g0dU-IqMkU7-axSqKfQPEsjTmfyNuIRlnYfRTAeGKSTTa7LrjFY9HMxRpiOt3AW3VwE4G2znb6gNWFY4N4nHV-yL3_8cIOfTgkJ0GBFEiOSwn5pdS7GyB_7aPfokZRG0nHSsxtuo7LZ8yFRPMLMReqXYXkFB596k9Va03YCKiD12whay9XkGRkXtapzPyYVcFdryoBzdgfAYOQSlnHINnrQNtgpvnuXchhg_POUbahEn8yXzN3O5ImqFXjWY2UV1yve9GuYPVgPVX0Ex30OrcAhfUlPtN2OM"
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def encrypt_file(file_path):
    """Encrypt a file and return the encrypted file path."""
    with open(file_path, "rb") as f:
        file_data = f.read()
    encrypted_data = cipher.encrypt(file_data)

    encrypted_file_path = file_path + ".enc"
    with open(encrypted_file_path, "wb") as f:
        f.write(encrypted_data)

    return encrypted_file_path

def upload_to_dropbox(file_path, dropbox_folder="/Encrypted/"):
    """
    Encrypts and uploads a file to Dropbox.

    Args:
        file_path (str): Local file path to upload.
        dropbox_folder (str): Destination folder in Dropbox (default: "/Encrypted/").
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    encrypted_file_path = encrypt_file(file_path)
    dropbox_destination = f"{dropbox_folder}{os.path.basename(encrypted_file_path)}"

    with open(encrypted_file_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_destination, mode=dropbox.files.WriteMode("overwrite"))

    os.remove(encrypted_file_path)  # Clean up local encrypted file
    print(f"✅ File uploaded to Dropbox: {dropbox_destination}")
