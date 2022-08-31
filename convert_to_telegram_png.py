import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import typer
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

from log import set_up_logger

set_up_logger()

logger = logging.getLogger(__name__)


def main(
        asset_name: str, show: bool = False, segmentation: bool = False, save: bool = False
) -> None:
    logger.info(f"Processing {asset_name}")

    asset_path = Path(asset_name)

    if asset_path.exists():
        logger.info("Asset exists")

        asset = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)

        logger.info(f"asset size is {asset.shape}")

        y, x, channels = asset.shape

        if x >= y:
            new_x = 512
            new_y = int(y / x * 512)
        else:
            new_y = 512
            new_x = int(x / y * 512)

        logger.info(f"new_x, new_y -> ({new_x, new_y})")

        output_asset = cv2.resize(asset, (new_x, new_y))
        output_asset_name = asset_name.replace("original", "resized")

        logger.info("resized")

        if segmentation:
            output_asset_name = asset_name.replace("resized", "segmented")
            with mp_selfie_segmentation.SelfieSegmentation(
                    model_selection=0) as selfie_segmentation:
                results = selfie_segmentation.process(
                    cv2.cvtColor(output_asset, cv2.COLOR_BGR2RGB)
                )

                BG_COLOR = (0, 0, 0, 0)  # transparent
                MASK_COLOR = (255, 255, 255, 255)  # white

                condition = np.stack((results.segmentation_mask,) * 4, axis=-1) > 0.1
                # Generate solid color images for showing the output selfie segmentation mask.
                fg_image = np.zeros((new_y, new_x, 4), dtype=np.uint8)
                fg_image[:] = MASK_COLOR
                bg_image = np.zeros((new_y, new_x, 4), dtype=np.uint8)
                bg_image[:] = BG_COLOR
                output_asset = cv2.cvtColor(output_asset, cv2.COLOR_BGR2RGBA)
                output_image = np.where(condition, output_asset, bg_image)

                output_asset = output_image

                logger.info("segmented")

        if save:

            if segmentation:
                # convert back to BGRA
                output_asset = cv2.cvtColor(output_asset, cv2.COLOR_BGR2RGB)

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


if __name__ == "__main__":
    typer.run(main)
