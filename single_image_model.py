import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage



import colorsys
from matplotlib.patches import Polygon
from skimage.measure import find_contours


from DamageDetectionFlask.mask_rcnn_damage_detection.mrcnn import visualize
# from mrcnn import visualize


# import mrcnn.model as modellib
from DamageDetectionFlask.mask_rcnn_damage_detection.mrcnn import model as modellib

from DamageDetectionFlask.mask_rcnn_damage_detection import custom
from importlib import reload # was constantly changin the visualization, so I decided to reload it instead of notebook

basedir = os.path.abspath(os.path.dirname(__file__))
# Directory to save logs and trained model
MODEL_DIR = os.path.join(basedir, "logs")

config = custom.CustomConfig()

custom_WEIGHTS_PATH = os.path.join(basedir, 'mask_rcnn_car_0001.h5')

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()



DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax



def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, image_id=''):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    # ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        im_name_ls = image_id.split('.')
        img_name = 'mcsnn_' + im_name_ls[-2] + '.png'

        save_dir = os.path.join(basedir, 'templates/img', img_name)

        my_dpi = 50


        plt.savefig(save_dir, bbox_inches="tight", dpi=my_dpi)
        return save_dir

def get_image(img_path, image_name):


    # img_path = r"C:\Users\Koshin\PycharmProjects\DamageDetectionFlask\templates\img\car-accident-00001.jpg"
    # image_id = "car-accident-00001.jpg"
    polygon = [{'name': 'polygon', 'all_points_x': [263, 273, 272, 263], 'all_points_y': [134, 135, 141, 134]}]

    image = skimage.io.imread(img_path)
    height, width = image.shape[:2]

    dataset = custom.CustomDataset()
    dataset.add_class("car", 1, "car")

    dataset.add_image("car",
                          image_id=image_name,
                          path=img_path,
                          height = height,
                          width = width,
                          polygons = polygon)


    # Must call before using the dataset
    dataset.prepare()

    # print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


    reload(visualize)



    image_id = dataset.image_ids[0]
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)


    # Run object detection
    with graph.as_default():
        results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'],
    #                             title="Predictions")

    save_dir = display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], image_id=image_name)
    return save_dir


def load_mask_rcnn_model():

    global model, graph

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)

    # Load weights
    print("Loading weights ", custom_WEIGHTS_PATH)
    model.load_weights(custom_WEIGHTS_PATH, by_name=True)
    graph = tf.get_default_graph()

