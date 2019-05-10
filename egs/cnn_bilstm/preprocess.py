from os.path import join, dirname


def read(filename):
    with open(filename) as f:
        content = f.read().split("\n")
    return content


data_path = join(dirname(dirname(dirname(__file__))), "data", "vlsp2016")
corpus_path = join(dirname(__file__), "data")
for name in ["train.txt", "dev.txt", "test.txt"]:
    file_name = join(data_path, name)
    corpus_file = join(corpus_path, name)
    with open(corpus_file, "w") as f:
        data = read(file_name)
        for line in data:
            text = line.replace("\t", " ")
            f.write(text + "\n")
