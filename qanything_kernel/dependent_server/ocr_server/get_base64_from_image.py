import base64


def image_to_base64(file_path: str):
    img_np = open(file_path, 'rb').read()
    return base64.b64encode(img_np).decode("utf-8")
