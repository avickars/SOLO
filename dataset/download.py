import fiftyone as fo
import fiftyone.zoo as foz

classes = ['person', 'cow', 'sheep', 'dog', 'horse', 'cat', 'bird', 'truck', 'car']

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=200,
    classes=classes,
    label_types=["detections", "segmentations"],
)
#
# dataset = foz.load_zoo_dataset(
#     "coco-2017",
#     split="test",
#     classes=classes,
#     label_types=["detections", "segmentations"],
# )

# dataset = foz.load_zoo_dataset(
#     "coco-2017",
#     split="validation",
#     max_samples=100,
#     classes=["cat", "dog"],
#     label_types=["detections", "segmentations"],
# )

session = fo.launch_app(dataset)

session.wait()


