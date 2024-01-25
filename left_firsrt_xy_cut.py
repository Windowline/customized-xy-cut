import numpy as np

def left_first_xy_cut(boxes: np.ndarray):
    output = []
    _left_first_xy_cut(boxes, output)
    return output


def left_first_xy_cut_indexes(boxes: np.ndarray, doc_width=None):
    output_index = []
    indexes = [True] * len(boxes)
    _left_first_xy_cut_indexes(boxes, indexes, output_index, doc_width)
    return output_index


def _left_first_xy_cut_indexes(boxes: np.ndarray, indexes, output, doc_width=None):
    if sum(indexes) == 0 or len(output) == len(boxes):
        return

    if sum(indexes) == 1:
        output.extend(np.where(indexes)[0])
        return

    left_interval = _split_by_first_zero_gap(_projection(boxes[indexes], 0))

    if all(v is not None for v in left_interval):
        condition = (boxes[:, 2] <= ((left_interval[0] + left_interval[1]) / 2))
        _left_first_xy_cut_indexes(boxes, indexes & condition, output, doc_width)
        _left_first_xy_cut_indexes(boxes, indexes & ~condition, output, doc_width)
        return

    top_interval = _split_by_first_zero_gap(_projection(boxes[indexes], 1), min_gap=3)

    if all(v is not None for v in top_interval):
        condition = (boxes[:, 3] <= ((top_interval[0] + top_interval[1]) / 2))
        _left_first_xy_cut_indexes(boxes, indexes & condition, output, doc_width)
        _left_first_xy_cut_indexes(boxes, indexes & ~condition, output, doc_width)
        return

    remain_boxes = list()
    original_indexes = dict()
    for i, flag in enumerate(indexes):
        if flag:
            remain_boxes.append(boxes[i])
            original_indexes[len(remain_boxes) - 1] = i
    remain_indexes = list(range(len(remain_boxes)))

    if doc_width: # do with heuristic
        sorted_indexes = sorted(remain_indexes, key=lambda k: _get_distance_mh(remain_boxes[k][0], remain_boxes[k][1], 0, 0, _step(remain_boxes[k], doc_width)))
    else:
        sorted_indexes = sorted(remain_indexes, key=lambda k: (remain_boxes[k][0], remain_boxes[k][1]))

    original_sorted_indexes = [original_indexes[i] for i in sorted_indexes]
    output.extend(original_sorted_indexes)


def _left_first_xy_cut(boxes: np.ndarray, output):
    if len(boxes) == 0:
        return

    if len(boxes) == 1:
        output.append(boxes[0].tolist())
        return

    left_interval = _split_by_first_zero_gap(_projection(boxes, 0))

    if all(v is not None for v in left_interval):
        condition = boxes[:, 2] <= left_interval[1]
        _left_first_xy_cut(boxes[condition], output)
        _left_first_xy_cut(boxes[~condition], output)
        return

    top_interval = _split_by_first_zero_gap(_projection(boxes, 1), min_gap=2)

    if all(v is not None for v in top_interval):
        condition = boxes[:, 3] <= top_interval[1]
        _left_first_xy_cut(boxes[condition], output)
        _left_first_xy_cut(boxes[~condition], output)
        return

    boxes_for_sort = boxes.tolist()
    sorted_boxes = sorted(boxes_for_sort, key=lambda k: (boxes_for_sort[k][0], boxes_for_sort[k][1]))
    output.extend(sorted_boxes)

def _split_by_first_zero_gap(proj, min_gap=1):
    ret_intervals = []
    curr_interval = [None, None]

    for i in range(len(proj)):
        if proj[i] == 0 and curr_interval[0] is None:
            curr_interval[0] = i
        elif proj[i] >= 1 and curr_interval[0] is not None:
            curr_interval[1] = i - 1

        if all(v is not None for v in curr_interval):
            ret_intervals.append([curr_interval[0], curr_interval[1]])
            curr_interval[0] = None
            curr_interval[1] = None

    ret_intervals = [x for x in ret_intervals if (x[1] - x[0] + 1) >= min_gap]

    if len(ret_intervals) >= 2:  # ret_intervals[1:-1]
        return ret_intervals[1]
    else:
        return [None, None]

def _projection(boxes: np.array, axis) -> np.ndarray:
    length = np.max(boxes[:, axis::2])
    res = np.zeros(length, dtype=int)
    for start, end in boxes[:, axis::2]:
        res[start:end] += 1
    return res

def _step(bboxes, width):
    min_x = bboxes[0]
    min_x /= width
    return (0, 0.5) if min_x < 0.5 else (1, 1)


def _get_distance_mh(x1, y1, x2, y2, w):
    return (abs(x2 - x1) * w[0]) + (abs(y2 - y1) * w[1])

def split_by_first_zero_gap(proj, min_gap=1):
    ret_interval = [None, None]

    for i in range(len(proj)):
        if proj[i] >= 1 and ret_interval[0] is None:
            ret_interval[0] = i
        elif proj[i] == 0 and ret_interval[0] is not None:
            ret_interval[1] = i

        if all(v is not None for v in ret_interval):
            if ret_interval[1] - ret_interval[0] >= min_gap:
                return ret_interval
            else:
                ret_interval = [None, None]

    return ret_interval


def _step(bboxes, width):
    min_x = bboxes[0]
    min_x /= width
    return (0, 0.5) if min_x < 0.5 else (1, 1)


def _get_distance_mh(x1, y1, x2, y2, w):
    return (abs(x2 - x1) * w[0]) + (abs(y2 - y1) * w[1])