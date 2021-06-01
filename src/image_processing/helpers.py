def crop_box(image, boxes, enlargebox_px):
    cropped_boxes = []

    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # x, y, w, h = box
        # enlarge box a bit
        x -= enlargebox_px
        y -= enlargebox_px
        w += enlargebox_px * 2
        h += enlargebox_px * 2

        bottom_right_x = x + w
        bottom_right_y = y + h

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if bottom_right_x > image.shape[1]:
            bottom_right_x = image.shape[1]
        if bottom_right_y > image.shape[0]:
            bottom_right_y = image.shape[0]

        cropped_box = image[y:bottom_right_y, x:bottom_right_x]
        cropped_boxes.append(cropped_box)

    return cropped_boxes