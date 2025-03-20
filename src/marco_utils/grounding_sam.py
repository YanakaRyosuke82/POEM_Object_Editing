from autodistill.detection import CaptionOntology
from autodistill_grounded_sam_2 import GroundedSAM2

# Initialize the GroundedSAM model
base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            "object": "object",
        }
    ),
    model="Grounding DINO",
)

obj_class = "cup"
base_model.ontology = CaptionOntology(
    {
        obj_class: obj_class,
    }
)
import PIL
import numpy as np

PIL_image = PIL.Image.open("/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/breaking_point_dataset_23/output/sample_001/initial_image.png")

predictions = base_model.predict(PIL_image)
results_mask = predictions.mask[0].astype(np.uint8) * 255
PIL.Image.fromarray(results_mask).save("/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/AAAAAAAA.png")
print(predictions)
