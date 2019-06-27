#!/usr/bin/env python3

import collections

# import tensorflow as tf
# import tensorflow.contrib.graph_editor as ge


class Node:
    def __init__(self, memory=1, name=''):
        self.name = name
        self.memory = memory
        self.parents = set()
        self.children = set()
        self.is_checkpoint = False
        self.backward_index = None

    def __str__(self):
        return self.name + '\t' + str(self.memory) + '\t' + \
            'backward index: ' + str(self.backward_index) + '\t' + \
            'parents: ' + str(self.parents) + '\t' + \
            'children: ' + str(self.children)

    def __repr__(self):
        return self.name


def chain(nodes):
    for i in range(len(nodes) - 1):
        nodes[i].children.add(nodes[i+1])
        nodes[i+1].parents.add(nodes[i])


def topological_sort(last_node, visited=set()):
    if last_node in visited:
        return []

    result = []
    for parent in last_node.parents:
        result += topological_sort(parent, visited)
    result.append(last_node)
    return result


def compute_memory_usage(last_node):
    nodes = topological_sort(last_node)

    for node in nodes:
        node.backward_index = 0

    backward_num = 0

    for node in nodes:
        backward_num = max(backward_num, node.backward_index + 1)

        for child in node.children:
            if node.is_checkpoint:
                child.backward_index = max(
                    child.backward_index, node.backward_index + 1)
            else:
                child.backward_index = max(
                    child.backward_index, node.backward_index)

    memory_by_backward = [0.0] * backward_num

    for node in nodes:
        memory_by_backward[node.backward_index] += node.memory
        if node.is_checkpoint:
            for i in range(node.backward_index + 1, len(memory_by_backward)):
                memory_by_backward[i] += node.memory

    print(memory_by_backward)
    return max(memory_by_backward)

a = Node(1, 'a')
b = Node(2, 'b')
c = Node(4, 'c')
d = Node(8, 'd')
e = Node(16, 'e')

# a.is_checkpoint = True
# b.is_checkpoint = True
# c.is_checkpoint = True
d.is_checkpoint = True
# e.is_checkpoint = True

chain([a, b, c, d, e])
print(a, b, c, d, e, sep='\n')



print(compute_memory_usage(e))

