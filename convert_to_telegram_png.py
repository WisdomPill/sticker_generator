import logging
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import typer
from log import set_up_logger

mp_selfie_segmentation = mp.solutions.selfie_segmentation

set_up_logger()

logger = logging.getLogger(__name__)


# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE


def main(
        asset_name: str = typer.Argument(
            ..., help="Asset name to transform in sticker"
        ),
        show: bool = typer.Option(
            False, help="Show output in using matplotlib.plt"
        ),
        segmentation: bool = typer.Option(
            False, help="Apply selfie segmentation from mediapipe"
        ),
        save: bool = typer.Option(
            False, help="Save image to resized or segmented folder, "
                        "based on --segmentation option"
        )
) -> None:
    """
    Welcome to sticker generator, which creates telegram images for stickers,
    and if you ask him kindly will also use some "AI" segmentation magic to highlight the subject of the image
    """
    logger.info(f"Processing {asset_name}")

    asset_path = Path(asset_name)

    if asset_path.exists():
        logger.info("Asset exists")

        asset = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)

        logger.info(f"asset size is {asset.shape}")

        new_x, new_y = get_new_size(asset)

        output_asset = cv2.resize(asset, (new_x, new_y))
        output_asset_name = asset_name.replace("original", "resized")

        logger.info("resized")

        if segmentation:
            output_asset_name = asset_name.replace("original", "segmented")
            with mp_selfie_segmentation.SelfieSegmentation(
                    model_selection=0) as selfie_segmentation:
                results = selfie_segmentation.process(
                    cv2.cvtColor(output_asset, cv2.COLOR_BGR2RGB)
                )

                segmentation_mask = results.segmentation_mask
                eroded_condition = erode_segmented_subjects(new_x, new_y,
                                                            segmentation_mask)

                condition = np.stack((eroded_condition,) * 4, axis=-1) == 255
                # Generate solid color images for showing the output selfie segmentation mask.
                fg_image = np.zeros((new_y, new_x, 4), dtype=np.uint8)
                fg_image[:] = (255, 255, 255, 255)  # white

                bg_image = np.zeros((new_y, new_x, 4), dtype=np.uint8)
                bg_image[:] = (0, 0, 0, 0)  # transparent
                fg_image = np.where(condition, fg_image, bg_image)

                element = generate_structuring_element(8)

                fg_image = cv2.dilate(fg_image, element)

                fg_mask = cv2.cvtColor(fg_image, cv2.COLOR_RGBA2GRAY)

                fg_mask_condition = np.stack((fg_mask,) * 4, axis=-1) == 255

                output_asset = cv2.cvtColor(output_asset, cv2.COLOR_BGR2RGBA)
                output_image = np.where(condition, output_asset, fg_image)
                output_image = np.where(fg_mask_condition, output_image, bg_image)

                output_asset = output_image

                logger.info("segmented")

        if save:

            if segmentation:
                # convert back to BGRA
                output_asset = cv2.cvtColor(output_asset, cv2.COLOR_RGBA2BGRA)

            output_asset_path = Path(output_asset_name)
            output_asset_path_png = output_asset_path.with_suffix(".png")

            output_asset_path_png.parent.mkdir(parents=True, exist_ok=True)

            logger.info("created parent dirs")

            cv2.imwrite(str(output_asset_path_png), output_asset)
            logger.info(f"saved to {output_asset_path_png}")

        if show:
            if not segmentation:
                # if not segmented then convert back to RGBA
                rgb_output_asset = cv2.cvtColor(output_asset, cv2.COLOR_BGR2RGBA)
                plt.imshow(rgb_output_asset)
            else:
                # when segmented then image is already in RGBA
                plt.imshow(output_asset)

            plt.show()
    else:
        logger.error("Asset does not exist")


def get_new_size(
        asset: np.ndarray
) -> Tuple[int, int]:
    """
    Get new asset size keeping aspect ratio. Do so while not overflowing 512x512 image
    :param asset: image that was read with cv2.imread (numpy.ndarray)
    :return: new size tuple
    """
    y, x, channels = asset.shape
    if x >= y:
        new_x = 512
        new_y = int(y / x * 512)
    else:
        new_y = 512
        new_x = int(x / y * 512)

    logger.info(f"new_x, new_y -> ({new_x, new_y})")

    return new_x, new_y


def generate_structuring_element(size: int) -> np.ndarray:
    """
    Generate structuring element based on morph shape and size
    :param size: size of structuring element
    :return:
    """
    return cv2.getStructuringElement(
        morph_shape(0), (2 * size + 1, 2 * size + 1), (size, size)
    )


def erode_segmented_subjects(new_x, new_y, segmentation_mask):
    condition = np.stack((segmentation_mask,), axis=-1) > 0.1
    # Generate solid color images for showing the output selfie segmentation mask.
    fg_image = np.zeros((new_y, new_x, 1), dtype=np.uint8)
    fg_image[:] = 255
    bg_image = np.zeros((new_y, new_x, 1), dtype=np.uint8)
    bg_image[:] = 0
    fg_image = np.where(condition, fg_image, bg_image)
    element = generate_structuring_element(1)
    dilated_condition = cv2.erode(fg_image, element)
    return dilated_condition


if __name__ == "__main__":
    typer.run(main)
