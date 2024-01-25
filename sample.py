from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from left_firsrt_xy_cut import left_first_xy_cut
from left_firsrt_xy_cut import left_first_xy_cut_indexes
from xy_cut import recursive_xy_cut
from sample_configs import *


def _pil_draw_rect(draw, point1, point2, reading_order, color):
    draw.rectangle((point1, point2), outline=color, width=1)
    draw.text((point1[0] + 5, point1[1]), str(reading_order), font=ImageFont.load_default(), fill=color)


def _pil_draw_reading_order_line(draw, prev_bbox, curr_bbox):
    prev_min_x, prev_min_y, prev_max_x, prev_max_y = prev_bbox
    min_x, min_y, max_x, max_y = curr_bbox
    prev_center = ((prev_min_x + prev_max_x) / 2, (prev_min_y + prev_max_y) / 2)
    curr_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    draw.line([prev_center, curr_center], fill='red', width=2)

    def rot2d(vec, radian):
        x, y = vec[0], vec[1]
        nx = x * math.cos(radian) - y * math.sin(radian)
        ny = x * math.sin(radian) + y * math.cos(radian)
        return nx, ny

    def normalize(vec, fix_len):
        x, y = vec[0], vec[1]
        mag = math.sqrt(x ** 2 + y ** 2) + 0.00000001
        nx = (x / mag) * fix_len
        ny = (y / mag) * fix_len
        return nx, ny

    vec = prev_center[0] - curr_center[0], prev_center[1] - curr_center[1]
    a0 = normalize(rot2d(vec, math.radians(45)), 6.0)
    a1 = normalize(rot2d(vec, math.radians(-45)), 6.0)
    p0 = curr_center[0] + a0[0], curr_center[1] + a0[1]
    p1 = curr_center[0] + a1[0], curr_center[1] + a1[1]
    draw.line([curr_center, p0], fill='red', width=2)
    draw.line([curr_center, p1], fill='red', width=2)


def run(boxes, input_img_path, output_img_path, algorithm):
    random_boxes = np.array(boxes)
    np.random.shuffle(random_boxes)

    if algorithm == "left_fisrt_xy_cut":
        box_color = (0, 0, 255)
        #direct sort
        # sorted_boxes = left_first_xy_cut(np.asarray(random_boxes).astype(int))
        #index sort
        sorted_indexes = left_first_xy_cut_indexes(np.asarray(random_boxes).astype(int), 700)
        original_indexes = list(range(len(random_boxes)))
        print("sorted: ", sorted_indexes)
        assert len(set(sorted_indexes)) == len(set(original_indexes))
        sorted_boxes = random_boxes[np.array(sorted_indexes)].tolist()
    else:
        res = []
        box_color = (0, 255, 0)
        recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(boxes)), res)
        sorted_boxes = random_boxes[np.array(res)].tolist()

    if input_img_path is None:
        return

    image = Image.open(input_img_path)

    for i, bbox in enumerate(sorted_boxes):
        draw = ImageDraw.Draw(image)
        _pil_draw_rect(draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), reading_order=i, color=box_color)
        if i > 0:
            _pil_draw_reading_order_line(draw, sorted_boxes[i - 1], bbox)

    plt.rcParams["figure.dpi"] = 300
    plt.imshow(np.array(image))
    plt.savefig(output_img_path)


if __name__ == "__main__":
    # check output image
    run(test_boxes1, "test_images/test1.png", "result_left_fisrt_xy_cut/result1.png", "left_fisrt_xy_cut")
    run(test_boxes2, "test_images/test2.png", "result_left_fisrt_xy_cut/result2.png", "left_fisrt_xy_cut")
    run(test_boxes3, "test_images/test3.png", "result_left_fisrt_xy_cut/result3.png", "left_fisrt_xy_cut")

    # run(test_boxes1, "test_images/test1.png", "result_xy_cut/result1.png", "xy_cut")
    # run(test_boxes2, "test_images/test2.png", "result_xy_cut/result2.png", "xy_cut")
    # run(test_boxes3, "test_images/test3.png", "result_xy_cut/result3.png", "xy_cut")
    #
    # # check time
    # num_iter = 500
    # start_time = time.time()
    # for _ in range(num_iter):
    #     run(test_boxes1, None, None, "left_fisrt_xy_cut")
    #     run(test_boxes2, None, None, "left_fisrt_xy_cut")
    #     run(test_boxes3, None, None, "left_fisrt_xy_cut")
    # end_time = time.time()
    # print("left_fisrt_xy_cut time: ", end_time - start_time)
    #
    # start_time = time.time()
    # for _ in range(num_iter):
    #     run(test_boxes1, None, None, "xy_cut")
    #     run(test_boxes2, None, None, "xy_cut")
    #     run(test_boxes3, None, None, "xy_cut")
    # end_time = time.time()
    # print("xy_cut time: ", end_time - start_time)
