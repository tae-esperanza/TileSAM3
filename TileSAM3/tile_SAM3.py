import torch
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import numpy as np
import cv2
import math
from PIL import Image
from itertools import product
from tqdm import tqdm

# From segment-anything/utils/amg.py
# Generates layers of tiling
def generate_crop_boxes(im_size, n_layers, overlap_ratio):
    crop_boxes = []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    crop_boxes.append([0, 0, im_w, im_h])

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n))

        cw = crop_len(im_w, n, overlap)
        ch = crop_len(im_h, n, overlap)

        xs = [int((cw - overlap) * i) for i in range(n)]
        ys = [int((ch - overlap) * i) for i in range(n)]

        for x0, y0 in product(xs, ys):
            crop_boxes.append([
                x0,
                y0,
                min(x0 + cw, im_w),
                min(y0 + ch, im_h),
            ])

    return crop_boxes

# Fills holes in masks
def fill_holes(mask):
    m = (mask > 0).astype(np.uint8) * 255
    flood = np.pad(m, 1, mode="constant")
    cv2.floodFill(flood, None, (0, 0), 255)
    flood = flood[1:-1, 1:-1]
    holes = cv2.bitwise_not(flood) & (~m)
    return ((m | holes) > 0)

# Applies convex hull to iregular shaped masks
def convex_fill_if_needed(
    mask,
    ratio_thresh=1.25, # Ratio of perimeter of mask to convex hull
    min_area=1000, # Only apply to larger masks
):
    m = (mask > 0).astype(np.uint8)

    area = m.sum()
    if area < min_area:
        return m.astype(bool)

    contours, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return m.astype(bool)

    cnt = max(contours, key=cv2.contourArea)

    actual_perim = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    convex_perim = cv2.arcLength(hull, True)

    if convex_perim == 0:
        return m.astype(bool)

    ratio = actual_perim / convex_perim

    if ratio <= ratio_thresh:
        return m.astype(bool)

    hull_mask = np.zeros_like(m)
    cv2.fillConvexPoly(hull_mask, hull, 1)

    return hull_mask.astype(bool)

# Returns only the largest component of the mask
def largest_connected_component(mask):
    mask = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return mask

    # stats: [label, x, y, w, h, area]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    return (labels == largest_label).astype(np.uint8)

# bounding box of mask
def bbox_from_mask(mask):
    mask = mask.astype(bool)

    if not mask.any():
        return None

    ys, xs = np.where(mask)

    x1 = xs.min()
    y1 = ys.min()
    x2 = xs.max() + 1
    y2 = ys.max() + 1

    return x1, y1, x2, y2

#determines if bounding box is on an internal edge of a tile
def bbox_on_internal_crop_edge(bbox, crop_box, orig_shape, pad=3):
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = crop_box
    H, W = orig_shape

    cw = cx1 - cx0
    ch = cy1 - cy0

    return (
        ((x0 <= pad) and cx0 > pad) or
        ((x1 >= cw - pad) and cx1 < W - pad) or
        ((y0 <= pad) and cy0 > pad) or
        ((y1 >= ch - pad) and cy1 < H - pad)
    )

# bounding box nms
# mask nms is very slow
def nms_boxes(boxes, scores, iou_thresh=0.5):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= iou_thresh]

    return torch.tensor(keep)

# masks contained in larger masks are filtered
def suppress_contained_streaming(results, area_tol=1.0):
    N = len(results)

    # Extract boxes and areas
    boxes = np.array([r["box"] for r in results])
    areas = np.array([r["mask"].sum() for r in results])

    # Sort by area descending
    order = np.argsort(-areas)

    keep = []
    suppressed = np.zeros(N, dtype=bool)

    x1, y1, x2, y2 = boxes.T

    for i in order:
        if suppressed[i]:
            continue

        keep.append(i)

        # Find candidates using bbox containment
        candidates = (
            (x1 >= x1[i]) &
            (y1 >= y1[i]) &
            (x2 <= x2[i]) &
            (y2 <= y2[i]) &
            (~suppressed) &
            (areas <= areas[i])
        )

        if not candidates.any():
            continue

        mask_i = results[i]["mask"]
        cand_indices = np.where(candidates)[0]

        # Filter with mask iou
        for cand_idx in cand_indices:
            if suppressed[cand_idx]:
                continue

            mask_cand = results[cand_idx]["mask"]
            intersection = (mask_cand & mask_i).sum()

            if intersection >= area_tol * areas[cand_idx]:
                suppressed[cand_idx] = True

        suppressed[i] = False

    return keep

# crops semantic mask of all detections
def crop_coverage(coverage, crop_box=None):
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
    else:
        x0, y0, x1, y1 = 0, 0, coverage.shape[1], coverage.shape[0]
    nz = np.count_nonzero(coverage[y0:y1, x0:x1])
    return nz / ((x1 - x0) * (y1 - y0))

# SAM3 inference on a single tile
def SAM3_on_tile(processor, image, crop_box, prompt, pad, scale=1):
    W, H = image.size
    x0, y0, x1, y1 = crop_box

    crop_im = image.crop((x0, y0, x1, y1))
    crop_w, crop_h = crop_im.size

    # Upscale crop to increase resolution
    # unnecessary
    up_w = crop_w * scale
    up_h = crop_h * scale
    crop_im_up = crop_im.resize((up_w, up_h), resample=Image.BICUBIC)

    state = processor.set_image(crop_im_up)
    out = processor.set_text_prompt(state=state, prompt=prompt)

    masks = out["masks"]    # [N,1,up_h,up_w]
    boxes = out["boxes"]    # local to upscaled crop
    scores = out["scores"]

    if masks is None or masks.numel() == 0:
        return None

    masks = masks[:, 0].float()  # [N,up_h,up_w]

    masks = F.interpolate(
        masks.unsqueeze(1),
        size=(crop_h, crop_w),
        mode="bicubic",
        align_corners=False
    )[:, 0]

    masks = masks > 0.5   # back to bool

    boxes = boxes / scale

    keep = []
    for i in range(len(boxes)):
        if not bbox_on_internal_crop_edge(
            boxes[i].cpu().numpy(),
            crop_box,
            (H, W),
            pad
        ):
            keep.append(i)

    if not keep:
        return None

    masks = masks[keep].cpu()
    boxes = boxes[keep].cpu()
    scores = scores[keep].cpu()

    boxes += torch.tensor([x0, y0, x0, y0])

    return {
        "masks": masks,      # crop-local, original resolution
        "boxes": boxes,      # global coordinates
        "scores": scores,
        "crop_box": crop_box,
    }

# Repeats SAM3 inference across hierarchical layers of tiling
def tileSAM3(
    processor,
    image,
    prompt,
    layers=5,
    overlap=0.25,
    nms_iou=0.5,
    pad=3,
    resize_scale = 1,
):
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)

    W, H = image.size
    coverage = np.zeros((H, W), dtype=np.uint8)

    crop_boxes = generate_crop_boxes((H, W), layers, overlap)

    detections = []

    pbar = tqdm(crop_boxes, desc="SAM3 tiling", leave=False)

    for crop_box in pbar:
        x0, y0, x1, y1 = crop_box

        # Skips tiles that have more coverage than the rest of image
        # if coverage[y0:y1, x0:x1].mean() > coverage.mean():
        #     continue

        out = SAM3_on_tile(processor, image, crop_box, prompt, pad, resize_scale)
        if out is None:
            continue

        tile_masks = out["masks"]
        tile_boxes = out["boxes"]
        tile_scores = out["scores"]

        for m, box, score in zip(tile_masks, tile_boxes, tile_scores):
            coverage[y0:y1, x0:x1] |= m.numpy().astype(np.uint8)

            detections.append({
                "mask": m.numpy().astype(np.uint8),
                "box": box,
                "score": score.item(),
                "crop_box": crop_box,
            })

        del out

    if not detections:
        return dict(
            masks=np.zeros((0, H, W), np.uint8),
            boxes=np.zeros((0, 4)),
            scores=np.zeros(0),
        )

    boxes = torch.stack([d["box"] for d in detections])
    scores = torch.tensor([d["score"] for d in detections])

    keep = nms_boxes(boxes, scores, nms_iou)

    final_results = []

    for i in keep:
        det = detections[i]
        x0, y0, x1, y1 = det["crop_box"]

        m_global = np.zeros((H, W), dtype=bool)

        m_local = det["mask"]
        m_local = largest_connected_component(m_local)
        m_local = fill_holes(m_local)
        m_local = convex_fill_if_needed(m_local)

        m_global[y0:y1, x0:x1] = m_local

        bbox = bbox_from_mask(m_global)

        if bbox is not None:
            final_results.append({
                "mask": m_global,
                "box": bbox,
                "score": det["score"]
            })

    if not final_results:
        return {
            "masks": np.zeros((0, H, W), np.uint8),
            "boxes": np.zeros((0, 4)),
            "scores": np.zeros(0),
        }

    # Remove contained objects
    keep2 = suppress_contained_streaming(final_results, area_tol=0.98)

    final_masks_arr = np.stack([final_results[i]["mask"] for i in keep2])
    final_boxes_arr = np.stack([final_results[i]["box"] for i in keep2])
    final_scores_arr = np.array([final_results[i]["score"] for i in keep2])

    return {
        "masks": final_masks_arr,
        "boxes": final_boxes_arr,
        "scores": final_scores_arr,
    }
