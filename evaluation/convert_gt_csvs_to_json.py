import os
import glob
import json
import argparse
import cv2


def main(dataset_root_path, subset, mask=False):
    """ Convert csv annotations into a single JSON and save it,
        to match the format with the algorithm submission output.

    Parameters:
    -----------
    dataset_root_path: str
        path to the dataset. (e.g. /home/user/ocelot2023_v0.1.1)

    subset: str
        `train` or `val` or `test`.
    """

    assert os.path.exists(f"{dataset_root_path}/annotations/{subset}")
    gt_paths = sorted(
        glob.glob(f"{dataset_root_path}/annotations/{subset}/cell/*.csv"))
    num_images = len(gt_paths)

    gt_json = {
        "type": "Multiple points",
        "num_images": num_images,
        "points": [],
        "version": {
            "major": 1,
            "minor": 0,
        }
    }
    meta_data = None
    with open(os.path.join(dataset_root_path, 'metadata.json'), 'r') as f:
        meta_data = json.load(f)['sample_pairs']

    for idx, gt_path in enumerate(gt_paths):
        filename = os.path.splitext(os.path.basename(gt_path))[0]
        tissue = None
        if mask:
            offset_x = int(
                (meta_data[filename]['patch_x_offset'] - 0.125) * 1024)
            offset_y = int(
                (meta_data[filename]['patch_y_offset'] - 0.125) * 1024)
            tissue = cv2.imread(
                f"{dataset_root_path}/annotations/{subset}/tissue/{filename}.png", 0)
            tissue = tissue[offset_y: offset_y + 256, offset_x: offset_x + 256]
            tissue = cv2.resize(tissue, (1024, 1024),
                                interpolation=cv2.INTER_NEAREST)
        with open(gt_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            x, y, c = line.split(",")
            x = int(x)
            y = int(y)
            if mask:
                if tissue[y, x] != 2:
                    continue
            point = {
                "name": f"image_{idx}",
                "point": [int(x), int(y), int(c)],
                "probability": 1.0,  # dummy value, since it is a GT, not a prediction
            }
            gt_json["points"].append(point)

    save_name = ''
    if mask:
        save_name = f"cell_gt_{subset}_mask.json"
    else:
        save_name = f"cell_gt_{subset}.json"
    with open(save_name, "w") as g:
        json.dump(gt_json, g)
        print(f"JSON file saved in {os.getcwd()}/{save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root_path", type=str, required=True,
                        help="Path to the dataset. (e.g. /home/user/ocelot2023_v0.1.1)")
    parser.add_argument("-s", "--subset", type=str, required=True,
                        choices=["train", "val", "test"],
                        help="Which subset among (trn, val, test)?")
    parser.add_argument("--mask", default=False,
                        required=False, action='store_true', help='mask')
    args = parser.parse_args()
    main(args.dataset_root_path, args.subset, args.mask)
