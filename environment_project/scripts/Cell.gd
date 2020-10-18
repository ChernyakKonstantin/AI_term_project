extends Control
""" 
This class provides a cell of grid world.
"""
class_name Cell

var block_name: int
var children_names: Array
var step: Vector3
var restricted: Array
var changed: bool = false

func _init() -> void:
	pass
	
func add(name: int, children: Array, block_step: Vector3, restricted_names: Array) -> void:
	"""
	The method sets the cell with the given attributes.
	Input: name - int,
		   children - Array,
		   block_step - Vector3,
		   restricted_names - Array.
	"""
	block_name = name
	children_names = children.duplicate(true)
	step = block_step
	restricted = restricted_names.duplicate(true)
	changed = true

func restrict(names: Array) -> void:
	"""
	The method sets given names of block types as restricted if
	the writting occurs the first time. Otherwise does nothing.
	Input: names - Array.
	"""
	if changed == false:
		restricted = names
		changed = true
		


