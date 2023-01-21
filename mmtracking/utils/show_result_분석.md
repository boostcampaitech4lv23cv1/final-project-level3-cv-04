### predictions  
```
# det_bboxes [tl_x, tl_y, br_x, br_y, score]
# [array([[6.8857770e+02, 3.5189542e+02, 1.3013917e+03, 1.2718206e+03,
#         9.4419754e-01],
#        [1.0364116e+03, 4.1763571e+02, 1.4457592e+03, 1.1393883e+03,
#         8.6316597e-01],
#        [5.8343445e+02, 4.0502762e+02, 9.4749817e+02, 1.1362501e+03,
#         8.5489428e-01],
#        [7.7634387e+02, 3.1212894e+02, 1.0529957e+03, 1.1307683e+03,
#         1.2197158e-02]], dtype=float32)]

# track_bboxes [id, tl_x, tl_y, br_x, br_y, score]
# [array([[5.00000000e+00, 6.88577698e+02, 3.51895416e+02, 1.30139172e+03,
#         1.27182056e+03, 9.44197536e-01],
#        [2.00000000e+00, 1.03641162e+03, 4.17635712e+02, 1.44575916e+03,
#         1.13938831e+03, 8.63165975e-01],
#        [4.00000000e+00, 5.83434448e+02, 4.05027618e+02, 9.47498169e+02,
#         1.13625012e+03, 8.54894280e-01]])]
```
  
### mmtrack>apis>models>mot>base.py  
```python
    def show_result(self,
                    img,
                    result,
                    score_thr=0.0,
                    thickness=1,
                    font_scale=0.5,
                    show=False,
                    out_file=None,
                    wait_time=0,
                    backend='cv2',
                    **kwargs):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                - The value of key 'track_bboxes' is list with length
                num_classes, and each element in list is ndarray with
                shape(n, 6) in [id, tl_x, tl_y, br_x, br_y, score] format.
                - The value of key 'det_bboxes' is list with length
                num_classes, and each element in list is ndarray with
                shape(n, 5) in [tl_x, tl_y, br_x, br_y, score] format.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_bboxes = result.get('track_bboxes', None)
        track_masks = result.get('track_masks', None)
        if isinstance(img, str):
            img = mmcv.imread(img)
        outs_track = results2outs(
            bbox_results=track_bboxes,
            mask_results=track_masks,
            mask_shape=img.shape[:2])
        img = imshow_tracks(
            img,
            outs_track.get('bboxes', None),
            outs_track.get('labels', None),
            outs_track.get('ids', None),
            outs_track.get('masks', None),
            classes=self.CLASSES,
            score_thr=score_thr,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            out_file=out_file,
            wait_time=wait_time,
            backend=backend)
        return img

```  
  
det_bboxes는 shape(n, 5) in [tl_x, tl_y, br_x, br_y, score] format.  
track_bboxes는 shape(n, 6) in [id, tl_x, tl_y, br_x, br_y, score] format.  
  
### mmtrack>core>utils>visualization.py  
```python
def _cv2_show_tracks(img,
                     bboxes,
                     labels,
                     ids,
                     masks=None,
                     classes=None,
                     score_thr=0.0,
                     thickness=2,
                     font_scale=0.4,
                     show=False,
                     wait_time=0,
                     out_file=None):
    """Show the tracks with opencv."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert ids.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 5
    if isinstance(img, str):
        img = mmcv.imread(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    inds = np.where(bboxes[:, -1] > score_thr)[0]
    bboxes = bboxes[inds]
    labels = labels[inds]
    ids = ids[inds]
    if masks is not None:
        assert masks.ndim == 3
        masks = masks[inds]
        assert masks.shape[0] == bboxes.shape[0]

    text_width, text_height = 9, 13
    for i, (bbox, label, id) in enumerate(zip(bboxes, labels, ids)):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = random_color(id)
        bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # score
        text = '{:.02f}'.format(score)
        if classes is not None:
            text += f'|{classes[label]}'
        width = len(text) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            text, (x1, y1 + text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # id
        text = str(id)
        width = len(text) * text_width
        img[y1 + text_height:y1 + 2 * text_height,
            x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            str(id), (x1, y1 + 2 * text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # mask
        if masks is not None:
            mask = masks[i].astype(bool)
            mask_color = np.array(bbox_color, dtype=np.uint8).reshape(1, -1)
            img[mask] = img[mask] * 0.5 + mask_color * 0.5

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img

```