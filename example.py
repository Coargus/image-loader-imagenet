from cog_imagenet import CogImageNetDataloader

if __name__ == "__main__":
    data = CogImageNetDataloader()
#     image_dir = "/store/datasets/ILSVRC"
#     classes = []
#     text = ""
#     for j, line in enumerate(open(image_dir + "/LOC_synset_mapping.txt")):
#         text += f"{j}. {line[9:].strip()} \n"
#     with open("imagenet_classes.txt", "w") as f:
#         f.write(text)
