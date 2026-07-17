from pathlib import Path
import sys

dchan = Path(__file__).resolve().parent.parent / "dchan"
sys.path.insert(0, str(dchan.resolve()))
from dchan.core.utils.stereopolis_orientation import DistortionPolynomial,STOrientation

if __name__ == "__main__":
    # distortion = DistortionPolynomial('./data/chantier_lilles/pvt_ori/Lille-150127_0485-11-00002_0000384.XML')
    ori_file = './data/chantier_lilles/pvt_ori/Lille-150127_0485-11-00002_0000384.XML'
    ori = STOrientation(ori_file)
    print(f"ori: {ori}")
    # undistort_tif(input_img, output_img, distortion)

#______________________________________
import os

# Mapping: container_prefix -> host env var
PATH_MAP = [
    ("/mast3r_ign/data/chantier_lilles", os.environ.get("HOST_DATA_CHANTIER_LILLES", "")),
    ("/mast3r_ign",                       os.environ.get("HOST_MAST3R_IGN_ROOT", "")),
]
