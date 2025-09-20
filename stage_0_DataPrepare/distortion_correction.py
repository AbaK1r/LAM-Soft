from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def process_file(pic_dir, output_dir_root, xmap, ymap):
    src = cv2.imread(str(pic_dir), cv2.IMREAD_UNCHANGED)
    dst = cv2.remap(src, xmap, ymap, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(str(output_dir_root / pic_dir.name), dst)


def prepare_image(image):
    # image = np.pad(image[36:1043, 700:1858], ((75, 76), (0, 0), (0, 0)), 'constant', constant_values=0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    return image


def main():
    new_w, new_h = 1024, 1024
    new_K = np.array([
        [0.51, 0., 0.5],
        [0., 0.51, 0.5],
        [0., 0., 1.]
    ], dtype=np.float64)
    new_K[0] *= new_w
    new_K[1] *= new_h
    print(new_K)

    with np.load('datasets/C3VD/C3VD_calibration.npz') as data:
        intrinsics: np.ndarray = data['K']
        dist: np.ndarray = data['D']
        print(intrinsics)
        xmap, ymap = cv2.fisheye.initUndistortRectifyMap(intrinsics, dist, None, new_K, (new_w, new_h), cv2.CV_16SC2)

    output_dir = Path('datasets/C3VD_processed')
    root_dir = Path('datasets/C3VD')
    folders = [folder for folder in root_dir.iterdir() if folder.is_dir()]

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = []
        for folder in folders:
            output_f = output_dir / folder.name
            output_f.mkdir(parents=True, exist_ok=True)
            image_ls = list(folder.glob('*'))
            for idx, img_name in enumerate(image_ls):
                file_name = img_name.name
                if not ('depth' in file_name or 'color' in file_name):
                    continue
                futures.append(executor.submit(process_file, img_name, output_f, xmap, ymap))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass


if __name__ == '__main__':
    main()
