import os
import re
import tempfile
import itertools
from functools import partial

import pandas as pd
import ast

from ..image_base import img_root_map
from .screenspot import ScreenSpot
from ..utils import build_judge, DEBUG_MESSAGE
from ...smp import *
from ...utils import track_progress_rich
from ipdb import set_trace as st

logger = get_logger("RUN")

"""
{
    "img_filename": "web_3b0ad239-da6b-4f6f-8f12-f674dc90ff33.png",
    "bbox": [42, 1102, 197, 70],
    "instruction": "view the details of the item",
    "data_type": "text",
    "data_source": "shop"
},
{
    "img_filename": "web_3b0ad239-da6b-4f6f-8f12-f674dc90ff33.png",
    "bbox": [93, 74, 86, 132],
    "instruction": "view the previous photo",
    "data_type": "icon",
    "data_source": "shop"
}
"""

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform pyautogui click/moveTo action to complete the task. The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`"""  # noqa: E501

USER_INSTRUCTION = """Please complete the following tasks by clicking using `pyautogui.click`:\n{instruction}"""  # noqa: E501

SYSTEM_PROMPT_V2 = """You are a GUI agent. You are given a screenshot of the screen and the description of a target element. You need to click the target element using `pyautogui.click`. The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`"""  # noqa: E501
USER_INSTRUCTION_V2 = """Please click the following target element using `pyautogui.click`:\n{description}"""


# Qwen3-VL specific prompts following mobile_agent.ipynb and computer_use.ipynb format
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/mobile_agent.ipynb
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb

# Mobile prompt - for ScreenSpot Mobile datasets
SYSTEM_PROMPT_QWEN3VL_MOBILE = """You are a helpful assistant that can use the "mobile_use" tool to interact with devices.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a device.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["click", "long_press", "scroll", "type", "back", "home"], "description": "The action to perform."}, "coordinate": {"type": "array", "items": {"type": "integer"}, "description": "The [x, y] coordinate to click/long_press/scroll, in range 0-999."}}, "required": ["action"]}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

# Computer prompt - for ScreenSpot Desktop/Web datasets
SYSTEM_PROMPT_QWEN3VL_COMPUTER = """You are a helpful assistant that can use the "computer_use" tool to interact with a computer.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer screen.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["left_click", "right_click", "double_click", "scroll", "type", "key", "move"], "description": "The action to perform."}, "coordinate": {"type": "array", "items": {"type": "integer"}, "description": "The [x, y] coordinate for mouse actions, in range 0-999."}}, "required": ["action"]}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

USER_INSTRUCTION_QWEN3VL = "Please click on the following element: {instruction}"


def parse_bbox_aguvis(response):
    match = re.search(r"x=([\d.]+), y=([\d.]+)", response)
    if match:
        click_point = [float(match.group(1)), float(match.group(2))]
    else:
        click_point = [0.0, 0.0]
    return click_point


def parse_bbox_qwen3vl_response(response):
    """
    Parse Qwen3-VL grounding response.

    Supports:
    1. Tool call format (mobile_use): <tool_call>{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [x, y]}}</tool_call>
    2. Tool call format (computer_use): <tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}</tool_call>

    Coordinates are in 0-999 range, returns center point normalized to 0-1.
    """
    import json

    # Try tool_call format
    tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    match = re.search(tool_call_pattern, response, re.DOTALL)

    if match:
        try:
            tool_call_json = match.group(1).strip()
            tool_call = json.loads(tool_call_json)
            coordinate = tool_call.get("arguments", {}).get("coordinate")
            if coordinate and len(coordinate) == 2:
                return [coordinate[0] / 1000.0, coordinate[1] / 1000.0]
        except json.JSONDecodeError:
            pass

    return []


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area

    return iou


def compute_accuracy(box1, box2, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    iou = compute_iou(box1, box2)
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    # Compute the center point of box 2
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2

    # Check if the center point is within box 1
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]


def convert_bbox(bbox, image_path):
    new_bbox = bbox if isinstance(bbox, list) else ast.literal_eval(bbox)
    new_bbox = [
        new_bbox[0],
        new_bbox[1],
        new_bbox[0] + new_bbox[2],
        new_bbox[1] + new_bbox[3],
    ]
    image = Image.open(image_path)
    img_size = image.size
    new_bbox = [
        new_bbox[0] / img_size[0],
        new_bbox[1] / img_size[1],
        new_bbox[2] / img_size[0],
        new_bbox[3] / img_size[1],
    ]
    return new_bbox


class ScreenSpotV2(ScreenSpot):
    MODALITY = "IMAGE"
    TYPE = "GUI"
    DATASET_URL = {
        "ScreenSpot_v2_Mobile": "ScreenSpot_v2_Mobile.tsv",
        "ScreenSpot_v2_Desktop": "ScreenSpot_v2_Desktop.tsv",
        "ScreenSpot_v2_Web": "ScreenSpot_v2_Web.tsv",
    }  # path
    DATASET_MD5 = {
        "ScreenSpot_v2_Mobile": "234c858ab4f0e787e8388a73df65a4b7",
        "ScreenSpot_v2_Desktop": "5f2aa2a497327bd33b2512a0c75cf994",
        "ScreenSpot_v2_Web": "01cd0877ee1b735a6d5190b053ba9482",
    }
    EVAL_TYPE = "point"  # point or rectangle
    RE_TYPE = "functional"  # type of referring expressions: functional or composite

    def __init__(
        self,
        dataset="ScreenSpot_Mobile",
        skip_noimg=True,
        skeleton=False,
        re_type="functional",
    ):
        # st()
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, "images", self.dataset_name)
        self.RE_TYPE = re_type
        if skeleton:
            return

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        data["index"] = [str(idx + 1) for idx, x in enumerate(data["bbox"])]

        self.meta_only = True
        self.parse_response_func = parse_bbox_aguvis  # TODO: parse function can be specified through kwargs when initializing the dataset # noqa: E501

        # The image field can store the base64 encoded image or another question index (for saving space)
        if "image" in data:
            data["image"] = [str(x) for x in data["image"]]
            image_map = {x: y for x, y in zip(data["index"], data["image"])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data["index"]]
            data["image"] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if "img_filename" in data:
            paths = [toliststr(x) for x in data["img_filename"]]
            data["image_path"] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data["index"]]):
            data["index"] = [int(x) for x in data["index"]]

        self.data = data
        self.post_build(dataset)

    def prepare_tsv(self, url, file_md5=None):
        if self.RE_TYPE == "functional":
            data_root = LMUDataRoot()
            data_path = osp.join(data_root, url)
        else:
            data_path = self.DATASET_URL_V2[self.dataset_name]
        return pd.DataFrame(load(data_path))

    def build_prompt(self, line):
        """Build prompt with Qwen3VL support via GROUNDING_MODEL environment variable."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)

        # Get model type from environment variable
        model_type = os.environ.get("GROUNDING_MODEL", None)

        if self.RE_TYPE == "functional":
            if model_type == "qwen3vl":
                user_instruction = USER_INSTRUCTION_QWEN3VL.format(
                    instruction=line["question"]
                )
            else:
                user_instruction = USER_INSTRUCTION.format(instruction=line["question"])
        else:
            user_instruction = USER_INSTRUCTION_V2.format(description=line["description"])

        msgs = []
        # add system prompt based on model type
        if model_type == "qwen3vl":
            # Select appropriate prompt based on dataset type (Mobile vs Desktop/Web)
            if "Mobile" in self.dataset_name:
                msgs.append(
                    dict(role="system", type="text", value=SYSTEM_PROMPT_QWEN3VL_MOBILE)
                )
            else:
                # Desktop and Web use computer_use tool
                msgs.append(
                    dict(
                        role="system", type="text", value=SYSTEM_PROMPT_QWEN3VL_COMPUTER
                    )
                )
        else:
            if self.RE_TYPE == "functional":
                msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT))
            else:
                msgs.append(dict(role="system", type="text", value=SYSTEM_PROMPT_V2))

        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs.append(dict(type="image", value=tgt_path))
        msgs.append(dict(type="text", value=user_instruction))
        return msgs

    def evaluate_point(self, eval_file, **judge_kwargs):
        """Evaluate with Qwen3VL parser support via GROUNDING_MODEL environment variable."""
        # -1: format_err, 0: wrong, 1: correct
        from collections import defaultdict
        stats = defaultdict(list)
        # Will include instance-level results
        result = []

        data = load(eval_file)
        assert "bbox" in data and "prediction" in data
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        # Get model type from environment variable
        model_type = os.environ.get("GROUNDING_MODEL", None)

        for i in tqdm(range(len(lines))):
            line = lines[i]
            bbox = (
                line["bbox"]
                if isinstance(line["bbox"], list)
                else ast.literal_eval(line["bbox"])
            )
            # The format of bbox is (x1, y1, w, h) for ScreenSpot v2
            x1, y1, w, h = bbox
            bbox = (x1, y1, x1 + w - 1, y1 + h - 1)

            img_name = line["image_path"]
            if not img_name.lower().endswith('.png'):
                img_name = img_name + '.png'
            image = Image.open(os.path.join(self.img_root, img_name))
            img_size = image.size

            def make_safe(value):
                if value == -1:
                    # we can tolerate -1 as a special value and normalize it to 0
                    return 0
                else:
                    return value

            bbox = [
                make_safe(bbox[0]) / img_size[0],
                make_safe(bbox[1]) / img_size[1],
                make_safe(bbox[2]) / img_size[0],
                make_safe(bbox[3]) / img_size[1],
            ]

            if any([x < 0 or x > 1 for x in bbox]):
                raise ValueError(
                    f"bbox out of range: {bbox} | {line['bbox']} | {img_size}"
                )

            key = (
                line["data_type"]
                if "category" not in line
                else line["category"] + ":" + line["data_type"]
            )
            prediction = str(line["prediction"])

            try:
                # Use Qwen3VL parser if specified
                if model_type == "qwen3vl":
                    click_point = parse_bbox_qwen3vl_response(prediction)
                    # Qwen3VL parser already returns normalized coordinates (0-1)
                    if not click_point:
                        # Fallback to aguvis parser
                        click_point = parse_bbox_aguvis(prediction)
                        if click_point[0] > 1 or click_point[1] > 1:
                            click_point = (click_point[0] / img_size[0], click_point[1] / img_size[1])
                else:
                    click_point = parse_bbox_aguvis(prediction)
                    # Do Normalization By Default
                    if click_point[0] > 1 or click_point[1] > 1:
                        click_point = (click_point[0] / img_size[0], click_point[1] / img_size[1])

                match = (bbox[0] <= click_point[0] <= bbox[2]) and (
                    bbox[1] <= click_point[1] <= bbox[3]
                )

                if match:
                    stats[key].append(1)
                else:
                    stats[key].append(0)
                is_wrong_format = False

            except Exception as e:
                logger.warning(f"exception in screenspot_v2 eval:{e}")
                stats[key].append(-1)
                match, is_wrong_format, click_point = False, True, None

            result.append(
                {
                    "img_path": os.path.join(self.img_root, line.get("image_path", str(line["index"]) + ".jpg")),
                    "text": line.get("instruction", line.get("question", "")),
                    "bbox": line["bbox"],
                    "parsed_bbox": bbox,
                    "type": line["data_type"],
                    "source": line.get("data_source", ""),
                    "match": match,
                    "is_wrong_format": is_wrong_format,
                    "pred": click_point,
                }
            )

        final_score_dict = {}
        # Record the number of each category
        final_score_dict.update({k + ':cnt': len(stats[k]) for k in stats})
        # Calculate the Overall stats
        full_stats = []
        for v in stats.values():
            full_stats.extend(v)
        final_score_dict['Overall_Accuracy'] = np.mean([x > 0 for x in full_stats]) * 100
        final_score_dict['Format_Err_Rate'] = np.mean([x < 0 for x in full_stats]) * 100
        # Calculate the Accuracy of Text / Icon
        text_stats = [v for k, v in stats.items() if k.endswith("text") for x in v]
        text_stats = itertools.chain(*text_stats)
        final_score_dict['Text_Accuracy'] = np.mean([x > 0 for x in text_stats]) * 100
        icon_stats = [v for k, v in stats.items() if k.endswith("icon") for x in v]
        icon_stats = itertools.chain(*icon_stats)
        final_score_dict['Icon_Accuracy'] = np.mean([x > 0 for x in icon_stats]) * 100
        # Calculate the Accuracy of Each Category
        if 'category' in data:
            cates = list(set(data['category']))
            for c in cates:
                sub_stats = [v for k, v in stats.items() if k.split(":")[0] == c for x in v]
                sub_stats = itertools.chain(*sub_stats)
                final_score_dict[c + '_Accuracy'] = np.mean([x > 0 for x in sub_stats]) * 100

        score_pth = eval_file.replace(".xlsx", "_score.json")
        dump(final_score_dict, score_pth)

        failure_cases_path = os.environ.get("FAILURE_CASES_PATH", None)
        if failure_cases_path is not None:
            def click_distance(bbox, click_point):
                x, y = click_point
                x1, y1, x2, y2 = bbox
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                abs_shift_to_center = [abs(x - xc), abs(y - yc)]
                width_outside, height_outside = [
                    max(0, abs_shift_to_center[0] - w / 2),
                    max(0, abs_shift_to_center[1] - h / 2),
                ]
                return (width_outside ** 2 + height_outside ** 2) ** 0.5

            wrong_format_result = [res for res in result if res["is_wrong_format"]]
            missed_result = [
                res for res in result if not res["match"] and not res["is_wrong_format"]
            ]
            missed_result.sort(
                key=lambda r: click_distance(r["parsed_bbox"], r["pred"]), reverse=True
            )
            failure_cases = wrong_format_result + missed_result

            with open(failure_cases_path, "w") as f:
                json.dump(failure_cases, f, indent=4, ensure_ascii=False)
        return final_score_dict
