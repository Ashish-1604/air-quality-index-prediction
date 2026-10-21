"""Download city_day.csv from Kaggle's public India air-quality dataset."""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
URL = "https://www.kaggle.com/api/v1/datasets/download/rohanrao/air-quality-data-in-india"
DESTINATION = ROOT / "kaggle_city_day.csv"


def main() -> None:
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0 AQI-Model/1.0"})
    with urlopen(request, timeout=120) as response:
        archive_bytes = response.read()
    import io
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = archive.getinfo("city_day.csv")
        with archive.open(member) as source, DESTINATION.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
    print(f"Downloaded {member.file_size:,} bytes to {DESTINATION}")


if __name__ == "__main__":
    main()
