import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class FileSaver():
    def __init__(self, file_name, path='./'):
        self.path = path
        self.file_name = file_name

    def write_header(self, header):
         with open(f"{self.path}/{self.file_name}", "w") as wf:
            wf.write(header)


    def append_content(self, content):
        with open(f"{self.path}/{self.file_name}", "a") as wf:
            wf.write(content)

    def write_content(self, content):
        with open(f"{self.path}/{self.file_name}", "w") as wf:
            wf.write(content)
