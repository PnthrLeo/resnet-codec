from src.model import ResNet18AE
import numpy as np
import skimage
import argparse


def load_from_binary(model_path, bin_path, image_path):
    model = ResNet18AE.load_from_checkpoint(model_path)

    model.eval()

    img = None
    for quality in range(1, 7):
        try:
            model.precision = quality
            img = model.decode_from_file(bin_path)
            break
        except AssertionError:
            pass
    img = img[0].permute(1, 2, 0).detach().numpy() * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    img = np.clip(img, 0, 1)
    
    skimage.io.imsave(image_path, img)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--bin_path", type=str, required=True)
    args.add_argument("--image_path", type=str, required=True)
    args.add_argument("--model_path", type=str, required=True)
    args = args.parse_args()
    
    load_from_binary(args.model_path, args.bin_path, args.image_path)
