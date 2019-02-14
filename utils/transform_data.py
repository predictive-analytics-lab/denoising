"""Transform the Huawei dataset"""
from pathlib import Path
import csv
from random import seed
from torchvision import transforms
F = transforms.functional
from tqdm import tqdm
from utils.loader import HuaweiDataset
import argparse

IMAGE_HEIGHT = 3968
IMAGE_WIDTH = 2976

def main(args: argparse.Namespace) -> None:
    rand_crop = transforms.RandomCrop(size=args.size)
    transform = transforms.Compose([rand_crop])

    if args.old_path is not None:
        root_path = Path(args.old_path).resolve()
    else:
        # Use the default path, assumes repo was cloned alongside a `data` folder
        root_path = Path(__file__).resolve().parent.parent.parent / "data"
    if not root_path.is_dir():
        raise ValueError("No valid data directory specified")

    if args.new_path is not None:
        transformed_path = Path(args.new_path).resolve()
    else:
        # Adds transformed folder to existing data dir
        transformed_path = root_path / "transformed"

    transformed_path.mkdir()
    data = HuaweiDataset(root_dir=root_path)
    dataset_info_writer = _get_dataset_info_writer(transformed_path)

    for image_no, sample in enumerate(tqdm(data)):
        clean_path = transformed_path / str(image_no) / "clean"
        clean_path.mkdir(parents=True)
        noisy_path = transformed_path / str(image_no) / "noisy"
        noisy_path.mkdir()

        patchsize = (args.size, args.size) if type(args.size) == int else tuple(args.size)

        patch_no = 0
        for stride in calculate_strides((IMAGE_HEIGHT, IMAGE_WIDTH), patchsize, args.overlap):
            patchsize = (args.size, args.size) if type(args.size) == int else tuple(args.size)
            with F.crop(sample['clean'], *stride, *patchsize) as clean:
                clean.save(clean_path / f"{patch_no}.png")
            with F.crop(sample['noisy'], *stride, *patchsize) as noisy:
                noisy.save(noisy_path / f"{patch_no}.png")
            patch_no += 1

        seed(image_no)

        for i in range(args.random_patches):
            clean = transform(sample['clean'])
            clean.save(clean_path / f"{patch_no+i}.png")
        seed(image_no)
        for i in range(args.random_patches):
            noisy = transform(sample['noisy'])
            noisy.save(noisy_path / f"{patch_no+i}.png")
        dataset_info_writer(noisy_path, clean_path, sample['iso'], sample['class'],
                            patch_no+args.random_patches)

    data.info_df.to_csv(str(transformed_path / "Training_Info.csv"), index=False)


def _get_dataset_info_writer(transformed_path):
    base_path = transformed_path.resolve()
    csv_path = base_path / "dataset.csv"
    with csv_path.open('w') as csv_file:  # reset CSV file by writing the heading
        csv.writer(csv_file).writerows([("noisy_path", "clean_path", "iso", "class")])

    def _write_dataset_info(noisy_path, clean_path, iso, image_class, patches):
        # relative paths
        noisy_rel = noisy_path.resolve().relative_to(base_path)
        clean_rel = clean_path.resolve().relative_to(base_path)
        dataset_info = [(noisy_rel / f"{i}.png", clean_rel / f"{i}.png", iso, image_class)
                        for i in range(patches)]
        with csv_path.open('a') as csv_file:  # append to the CSV file
            csv.writer(csv_file).writerows(dataset_info)

    return _write_dataset_info


def calculate_strides(imagesize, patchsize, overlap):
    # uses height * width in tuples (to match torch)
    if type(overlap) == int:
        overlap = (overlap, overlap)
    vertical_stride = patchsize[0] - (overlap[0] // 2)
    n_vertical_strides = -(-imagesize[0] // vertical_stride)
    horizontal_stride = patchsize[1] - (overlap[1] // 2)
    n_horizontal_stride = -(-imagesize[1] // horizontal_stride)

    strides = []
    for i in range(n_vertical_strides):
        for j in range(n_horizontal_stride):
            top = min(i*vertical_stride, imagesize[0] - patchsize[0])
            left = min(j*horizontal_stride, imagesize[1] - patchsize[1])
            strides.append((top, left))
    return strides


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate transformed images")
    parser.add_argument("size", metavar="s", type=int, help="Dimension of cropped image")
    parser.add_argument("--overlap", type=int, default=0, help="")
    parser.add_argument("--random-patches", type=int, default=0, help="Number of random crops per image")
    parser.add_argument("--data-path", dest="old_path", default=None, help="Data folder path")
    parser.add_argument("--new-path", dest="new_path", default=None,
                        help="Folder to store transformed data, must not exist yet")
    args = parser.parse_args()
    main(args)
