from src.model import ResNet18AE
from torchvision import transforms
import skimage
import argparse


def compress_image(model_path, image_path, bin_path, quality=1):
    model = ResNet18AE.load_from_checkpoint(model_path)

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = skimage.io.imread(image_path)
    img = transform(img)
    img = img.unsqueeze(0)
    model.precision = quality
    model.encode_to_file(img, bin_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--bin_path", type=str, required=True)
    parser.add_argument("--quality", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    compress_image(args.model_path, args.image_path, args.bin_path, args.quality)
