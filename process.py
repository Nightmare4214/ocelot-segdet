from util import gcio
from util.constants import (
    GC_CELL_FPATH,
    GC_TISSUE_FPATH,
    GC_METADATA_FPATH,
    GC_DETECTION_OUTPUT_PATH
)
from user.inference import Model


# from user.unet_example.unet import PytorchUnetCellModel as Model

def process(subset='test'):
    """Process a test patches. This involves iterating over samples,
    inferring and write the cell predictions
    """
    # Initialize the data loader
    loader = gcio.DataLoader(GC_CELL_FPATH, GC_TISSUE_FPATH)

    # Cell detection writer
    writer = gcio.DetectionWriter(GC_DETECTION_OUTPUT_PATH)

    # Loading metadata
    meta_dataset = gcio.read_json(GC_METADATA_FPATH)

    # Instantiate the inferring model
    model = Model(meta_dataset['sample_pairs'])

    pair_keys = []
    for k, v in meta_dataset['sample_pairs'].items():
        if v['subset'] == subset:
            pair_keys.append(k)
    pair_keys.sort()

    # NOTE: Batch size is 1
    for cell_patch, tissue_patch, pair_id in loader:
        print(f"Processing sample pair {pair_id}")
        # Cell-tissue patch pair inference
        cell_classification = model(
            cell_patch, tissue_patch, pair_keys[pair_id])

        # Updating predictions
        writer.add_points(cell_classification, pair_id)

    # Export the prediction into a json file
    writer.save()


if __name__ == "__main__":
    process('test')
