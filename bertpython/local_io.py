import os


class LocalIO(object):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def get_absolute_path(self) -> str:
        return "/".join(os.getcwd(), self.file_name)

    def load_csv(self) -> list:
        with open(self.file_name) as f:
            return [line.split(",") for line in f]

    def load_txt(self) -> str:
        with open(self.file_name) as f:
            return " ".join([line for line in f])