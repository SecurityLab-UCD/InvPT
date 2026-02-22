import typer
import os
import logging
import requests
import tarfile
from tqdm import tqdm


def get_fullname(subset: str) -> str:
    return f"Project_CodeNet_{subset}"


def get_tar_name(subset: str) -> str:
    return f"{get_fullname(subset)}.tar.gz"


def get_url(subset: str) -> str:
    tar_name = get_tar_name(subset)
    url = f"https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/{tar_name}"

    return url


def download(workdir: str, subset: str) -> str:
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    else:
        logging.warning(f"Directory {workdir} already exists")

    tar_path = os.path.join(workdir, get_tar_name(subset))
    if os.path.exists(tar_path):
        logging.warning(f"File {tar_path} already exists")
        return tar_path

    logging.info(f"Downloading {subset} dataset to {tar_path}")
    url = get_url(subset)
    response = requests.get(url, stream=True)
    # Download with a nice looking progress bar
    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(tar_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    return tar_path


def extract(workdir: str, tar_path: str):
    file_name = os.path.basename(tar_path).split(".")[0]
    ds_dir = os.path.join(workdir, file_name)
    if os.path.exists(ds_dir):
        logging.warning(f"Directory {ds_dir} already exists")
        return ds_dir

    logging.info(f"Extracting {tar_path} to {ds_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers()):
            tar.extract(member, path=workdir)

    return ds_dir


def main(subset: str, workdir: str = "./raw"):
    assert subset in ["C++1000", "C++1400", "Java250", "Python800"]

    tar_path = download(workdir, subset)
    ds_dir = extract(workdir, tar_path)
    logging.info(f"Dataset extracted to {ds_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(main)
