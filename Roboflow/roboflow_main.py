from roboflow import Roboflow

rf = Roboflow(api_key="9iqpq8QT76GKGjhDC5KM")
project = rf.workspace("srivalli-yada").project("road-vehicles-h3m1y")
version = project.version(1)
dataset = version.download("yolov8")