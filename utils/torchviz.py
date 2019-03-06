from graphviz import Digraph
import torch
from torch.autograd import Variable
from collections import namedtuple

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def make_dot(var, params=None):
	"""
	Prodice Graphviz representation of PyTorch autograd graph
	:param var:
	:param params:
	:return:
	"""
	if params is not None:
		assert all(isinstance(p, Variable) for p in params.values())
		params_map = {id(v): k for k, v in params.items()}
	node_attr = dict(style='filled',
					 shape='box',
					 align='left',
					 fontsize='12',
					 ranksep='0.1',
					 height='0.2')
	dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
	seen = set()

	def size2str(size):
		return '(' + (', ').join(['%d' % v for v in size]) + ')'

	def add_nodes(var):
		if var not in seen:
			if torch.is_tensor(var):
				dot.node(str(id(var)), size2str(var.size()), fillcolor='orange')
			elif hasattr(var, 'variable'):
				u = var.variable
				name = params_map[id(u)] if params is not None else ''
				node_name = '%s\n %s' % (name, size2str(u.size()))
				dot.node(str(id(var)), node_name, fillcolor='lightblue')
			else:
				dot.node(str(id(var)), str(type(var).__name__))
			seen.add(var)

			if hasattr(var, 'next_functions'):
				for u in var.next_functions:
					if u[0] is not None:
						dot.edge(str(id(u[0])), str(id(var)))
						add_nodes(u[0])
			if hasattr(var, 'saved_tensors'):
				for t in var.saved_tensors:
					dot.edge(str(id(t)), str(id(var)))
					add_nodes(t)

	add_nodes(var.grad_fn)
	resize_graph(dot)
	return dot


def resize_graph(dot, size_per_element=0.15, min_size=12):
	"""
	resize the grapth according to the content
	:param dot:
	:param size_per_element:
	:param min_size:
	:return:
	"""
	num_rows = len(dot.body)
	content_size = num_rows * size_per_element
	size = max(min_size, content_size)
	size_str = str(size) + "," + str(size)
	dot.graph_attr.update(size=size_str)
