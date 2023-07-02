import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Image Style Transfer")

    parser.add_argument("--content_image")
    parser.add_argument("--style_image")
    parser.add_argument("--save_dir", default="samples")
    parser.add_argument("--style_weight", type=int, default=1000)

    args = parser.parse_args()
    return args
