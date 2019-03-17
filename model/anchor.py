import numpy as np

input_size = [331,331]
_default_anchors_setting = (
	dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]), #7 *7
	dict(layer='p4', stride=32, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]), # 7*7
	dict(layer='p5', stride=64, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),# 4*4
)


def generator_default_anchor_maps(anchors_setting=None, input_shape=input_size):
	"""
	# TODO different anchor setting
	:param anchors_setting:
	:param input_shape:
	:return:
	"""
	if anchors_setting is None:
		anchors_setting = _default_anchors_setting
	center_anchors = np.zeros((0, 4), dtype=np.float32)
	edge_anchors = np.zeros((0, 4), dtype=np.float32)
	anchor_areas = np.zeros((0,), dtype=np.float32)
	input_shape=np.array(input_shape,dtype=np.float32)
	for anchor_info in anchors_setting:
		stride = anchor_info['stride']
		size = anchor_info['size']
		scales = anchor_info['scale']
		aspect_ratios = anchor_info['aspect_ratio']

		out_map_shape = np.ceil(input_shape/ stride)
		out_map_shape = out_map_shape.astype(np.int)
		output_shape = tuple(out_map_shape) + (4,)
		ostart = stride / 2.
		oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
		oy = oy.reshape(output_shape[0], 1)
		ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
		ox = ox.reshape(1, output_shape[1])

		center_anchors_map_template = np.zeros(output_shape, dtype=np.float32)
		center_anchors_map_template[:, :, 0] = oy
		center_anchors_map_template[:, :, 1] = ox

		for scale in scales:
			for aspect_ratio in aspect_ratios:
				center_anchors_map = center_anchors_map_template.copy()
				center_anchors_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
				center_anchors_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5

				edge_anchors_map = np.concatenate((center_anchors_map[..., :2] - center_anchors_map[..., 2:4] / 2.,
												   center_anchors_map[..., :2] + center_anchors_map[..., 2:4] / 2.),
												  axis=-1)
				anchor_area_map = center_anchors_map[..., 2] * center_anchors_map[..., 3]
				center_anchors = np.concatenate((center_anchors, center_anchors_map.reshape(-1, 4)))
				edge_anchors = np.concatenate((edge_anchors, edge_anchors_map.reshape((-1, 4))))
				anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))
	return center_anchors, edge_anchors, anchor_areas


def hard_nms(cdd, topn=10, iou_threshold=0.5):
	"""

	:param cdd: [score,x1,y1,x2,y2]
	:param topn:
	:param iou_threshold:
	:return:
	"""
	if not (type(cdd).__module__ == 'numpy' and len(cdd.shape) == 2 and cdd.shape[1] >= 5):
		raise TypeError('edge_box_map should be N*5 +ndarray')

	cdd = cdd.copy()
	indices = np.argsort(cdd[:, 0])
	cdd = cdd[indices]
	cdd_results = []
	res = cdd
	while res.any():
		cdd = res[-1]
		cdd_results.append(cdd)
		if len(cdd_results) == topn:
			return np.array(cdd_results)
		res = res[:-1]
		start_max = np.maximum(res[:, 1:3], cdd[1:3])
		end_min = np.minimum(res[:, 3:5], cdd[3:5])
		lengths = end_min - start_max
		intersec_map = lengths[:, 0] * lengths[:, 1]
		intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
		iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1]) * (res[:, 4] - res[:, 2]) + (
				cdd[3] - cdd[1]) * (cdd[4] - cdd[2]) - intersec_map)
		res = res[iou_map_cur < iou_threshold]
	return np.array(cdd_results)


if __name__ == '__main__':
	a = hard_nms(np.array([
		[0.4, 1, 10, 12, 20],
		[0.6, 1, 11, 11, 20],
		[0.55, 20, 30, 40, 50]]), topn=100, iou_threshold=0.4)
	print(a)
	generator_default_anchor_maps()