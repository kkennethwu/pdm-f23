from PIL import Image
import os
import natsort

def save_to_gif(object=None):
    root_dir = f"path_{object}"
    img_list = []
    for i in natsort.natsorted(os.listdir(root_dir)):
        if i.endswith(".jpg") and i.startswith("RGB"):
            img = Image.open(os.path.join(root_dir, i))
            img_list.append(img)
    # save as gif
    img_list[0].save(os.path.join(root_dir, "video.gif"), save_all=True, append_images=img_list[1:], duration=100, loop=0, quality=8)
    

if __name__ == "__main__":
    save_to_gif()