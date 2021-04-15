import cv2

def letterbox(img, new_shape: int = 640):
    """
    Resize image to a 32-pixel-multiple rectangle. Keep the proportions by adding constant padding
    :return: the resized image, the ratio tuple (r_height, r_width) where r_height=new_height / old_height, the pad
    tuple (p_vertical, p_horizontal)
    """
    padding_color = [114, 114, 114]

    # Make sure new_shape is correct
    assert new_shape / 32 == new_shape // 32

    height, width = img.shape[:2]

    # Scale ratio (new / old)
    ratio = min(new_shape / height, new_shape / width)

    # Compute padding
    new_unpad_width = int(round(width * ratio))
    new_unpad_height = int(round(height * ratio))
    padding_width = (new_shape - new_unpad_width)
    padding_height = (new_shape - new_unpad_height)

    if new_unpad_width != new_shape and new_unpad_width != new_shape:

        # Random resizing technique
        img = cv2.resize(img, (new_unpad_width, new_unpad_height))

    padding_top = padding_height // 2
    padding_bottom = padding_height - padding_height // 2
    padding_left = padding_width // 2
    padding_right = padding_width - padding_width // 2
    img = cv2.copyMakeBorder(img, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT,
                             value=padding_color)
    return img, (ratio, ratio), (padding_left, padding_top)