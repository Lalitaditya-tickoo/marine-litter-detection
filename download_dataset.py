from roboflow import Roboflow

rf = Roboflow(api_key="gZWQpzKMQPinaiHDxq6R") 

project = rf.workspace("marine-litter-water-segementation").project("underwater-marine-litter-instanc")

version = project.version(2)

dataset = version.download("coco-segmentation")
