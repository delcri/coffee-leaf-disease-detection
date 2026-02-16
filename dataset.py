from roboflow import Roboflow

rf = Roboflow(api_key="abcdef1234567890")  # tu API Key real
project = rf.workspace("jonatan-fragoso").project("dpcl-bracol-for-region-detect")
dataset = project.version(1).download("yolov8")
