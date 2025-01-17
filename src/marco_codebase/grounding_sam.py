import numpy as np
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam_2 import GroundedSAM2


class GroundedSAM2Predictor:
    def predict(self, input_image: np.ndarray, object_class: str) -> np.ndarray:
        self.base_model = GroundedSAM2(
            ontology=CaptionOntology({object_class: object_class}),
            model="Grounding DINO",
        )
        results = self.base_model.predict(input_image)
        # print(object_class, results)
        binary_mask = np.array(results.mask[0], dtype=np.uint8) * 255
        return binary_mask

    def __call__(self, input_image: np.ndarray, transform_dict: dict) -> dict:
        for obj, transforms in transform_dict.items():
            binary_mask = self.predict(input_image, obj)
            transforms["mask"] = binary_mask

        return transform_dict


if __name__ == "__main__":
    import cv2

    predictor = GroundedSAM2Predictor()
    image = np.array(cv2.imread("/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/data/src_image_marco/elephant_resized.png"))
    # mask = predictor.predict(image, "mirror")
    # cv2.imwrite("/dtu/blackhole/00/215456/Mask2Mask/container1_mask.jpg", mask)
    # print("Mask saved to /dtu/blackhole/00/215456/Mask2Mask/container1_mask.jpg")
    transform_dict = {
        "mirror": {"mask_transforms": [], "advanced_transforms": {"style": "monet"}},
        "text": {"mask_transforms": [], "advanced_transforms": {"style": "monet"}},
    }
    result = predictor(image, transform_dict)
    print(result)

    # save
    for obj, transforms in result.items():
        mask = transforms["mask"]
        cv2.imwrite(f"{obj}_mask.jpg", mask)