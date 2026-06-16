def preprocess(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_t, _ = preproc(img_rgb, None, INPUT_SIZE)

    return torch.from_numpy(img_t).unsqueeze(0).float().to(DEVICE)
def run_inference(img):

    h, w = img.shape[:2]

    tensor = preprocess(img)

    with torch.no_grad():

        outputs = model(tensor)

        print("Raw outputs:", outputs)

        outputs = postprocess(
            outputs,
            num_classes=NUM_CLASSES,
            conf_thre=CONF_THRESH,
            nms_thre=IOU_THRESH
        )

        print("After postprocess:", outputs)

    if outputs[0] is None:
        return []

    scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)

    detections = []

    for det in outputs[0].cpu().numpy():

        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det

        detections.append({
            "class_id": int(cls_id),
            "class": CLASS_NAMES[int(cls_id)],
            "confidence": round(float(obj_conf * cls_conf), 4),
            "bbox": [
                round(float(x1) / scale, 2),
                round(float(y1) / scale, 2),
                round(float(x2) / scale, 2),
                round(float(y2) / scale, 2)
            ]
        })

    return detections