from roboflow import Roboflow
rf = Roboflow(api_key="iTI466MWysEUT4sJzrX4")
project = rf.workspace("tulanelab").project("skin-lesion-detection-dnuq9")
version = project.version(4)
dataset = version.download("yolov5")