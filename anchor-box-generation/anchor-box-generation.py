import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    Returns list of [x1, y1, x2, y2].
    """

    # Handle int or tuple inputs
    if isinstance(feature_size, int):
        H = W = feature_size
    else:
        H, W = feature_size

    if isinstance(image_size, int):
        img_h = img_w = image_size
    else:
        img_h, img_w = image_size

    stride_h = img_h / H
    stride_w = img_w / W

    anchors = []

    for i in range(H):
        for j in range(W):
            cx = (j + 0.5) * stride_w
            cy = (i + 0.5) * stride_h

            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([x1, y1, x2, y2])

    return anchors   # ← list, not numpy array