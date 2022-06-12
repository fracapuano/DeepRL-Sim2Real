import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Warhol():
    def __init__(self, figure_path="./"):
        self.figure_path = figure_path

    def plot_column(self, x, y, title, axis_labels, figure_name, save=True):
        # fig, ax = plt.subplots()
        # ax.plot(x, y)
        # if logscale_bool:
        #     ax.set_xscale("log")
        # ax.set_xlabel(axis_labels[0])
        # ax.set_ylabel(axis_labels[1])
        # ax.set_ylim(bottom=-2, top=10)
        # ax.set_title(title)
        # plt.show()
        # if save:
        #     plt.savefig(f"./{self.figure_path}/{figure_name}.png")
        # plt.close(fig)
        pass

    def plot_columns(self, y, title, axis_labels, figure_name, save=True):
        # #plt.xscale("log")
        # plt.xlabel(axis_labels[0])
        # plt.ylabel(axis_labels[1])
        # plt.ylim(bottom=-2, top=10)
        # plt.title(title)
        # for column, i in zip(y, range(len(y))):
        #     plt.plot(column, label=f"{i}")
        # plt.legend()
        # plt.show()
        # if save:
        #     plt.savefig(f"./{self.figure_path}/{figure_name}.png")
        pass

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
