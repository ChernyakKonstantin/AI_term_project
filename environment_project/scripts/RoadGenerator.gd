extends Node
""" 
This class provides a random road generator.
The constant value n_blocks defines lenght of the road.
The constant value size_coeff is defined by road blocks scale 
during export from a 3D-editor.
"""

class_name RoadGenerator

#Define all constants
const size_coeff: float = 2.0

enum {L1, L2, L3, L4, R1, R2, R3, R4, S1, S2, S3, S4} #0 -> 11

#Define children for each type of blocks
const l1_children: Array = [L2, S4, R4]
const l2_children: Array = [L3, S3, R3]
const l3_children: Array = [L4, S2, R2]
const l4_children: Array = [L1, S1, R1]

const r1_children: Array = [L4, S2, R2]
const r2_children: Array = [L3, S3, R3]
const r3_children: Array = [L2, S4, R4]
const r4_children: Array = [L1, S1, R1]

const s1_children: Array = [L1, S1, R1]
const s2_children: Array = [L4, S2, R2]
const s3_children: Array = [L3, S3, R3]
const s4_children: Array = [L2, S4, R4]

const children: Array = [l1_children, l2_children, l3_children, l4_children,
						 r1_children, r2_children, r3_children, r4_children,
						 s1_children, s2_children, s3_children, s4_children
						]

#Define steps for each type of blocks.
const l1_step = Vector3(-1,0,0)
const l2_step = Vector3(0,0,1)
const l3_step = Vector3(1,0,0) 
const l4_step = Vector3(0,0,-1)

const r1_step = Vector3(1,0,0) 
const r2_step = Vector3(0,0,1)
const r3_step = Vector3(-1,0,0) 
const r4_step = Vector3(0,0,-1) 

const s1_step = Vector3(0,0,-1)
const s2_step = Vector3(1,0,0) 
const s3_step = Vector3(0,0,1)
const s4_step = Vector3(-1,0,0)

const steps: Array = [l1_step, l2_step, l3_step, l4_step,
					  r1_step, r2_step, r3_step, r4_step,
					  s1_step, s2_step, s3_step, s4_step
					]
#Define paths to scenes for each type of blocks
const l1_block = preload("res://assets/L1.tscn")
const l2_block = preload("res://assets/L2.tscn")
const l3_block = preload("res://assets/L3.tscn")
const l4_block = preload("res://assets/L4.tscn")

const r1_block = preload("res://assets/R1.tscn")
const r2_block = preload("res://assets/R2.tscn")
const r3_block = preload("res://assets/R3.tscn")
const r4_block = preload("res://assets/R4.tscn")

const s1_block = preload("res://assets/S1.tscn")
const s2_block = preload("res://assets/S2.tscn")
const s3_block = preload("res://assets/S3.tscn")
const s4_block = preload("res://assets/S4.tscn")

const blocks: Array = [l1_block, l2_block, l3_block, l4_block,
					   r1_block, r2_block, r3_block, r4_block,
					   s1_block, s2_block, s3_block, s4_block
					  ]

const terminal_block = preload("res://assets/Limit.tscn")

#Define restricted neighbour blocks for each type of blocks
const up_restrictions: Array = [L2, R2, S3]
const down_restrictions: Array = [L4, R4, S1] 
const left_restrictions: Array = [L3, R1, S2]
const right_restrictions: Array = [L1, R3, S4]
const all_restrictions: Array = [L1, L2, L3, L4, R1, R2, R3, R4, S1, S2, S3, S4]

#Define a grid parameters
const n_blocks: int = 31 #Grid is a square
const grid_size: int = 2 * n_blocks

#Define variables 
var rng = RandomNumberGenerator.new()
var pos: Vector3
var grid: Array

func _ready():
	_build()

func _build():
	"""
	The method build the road.
	"""
	var new_block: int 
	var complete: bool = true
	rng.randomize() #Enable randomness
	_init_grid()
	#Place the 1st block (s3) #Implement "statr limit" block
	new_block = S3
	#The 1st block is in the center of the grid
	pos.x = int(grid_size / 2)
	pos.z = int(grid_size / 2)
	grid[pos.z][pos.x].add(new_block, children[new_block], steps[new_block], all_restrictions)
	_new_node(terminal_block, "Start")
	_new_node(blocks[new_block], str(0))
	_add_restrictions(new_block)
	_update_pos(new_block)
	#Place the rest blocks
	for i in range(n_blocks - 1):
		var possible_names = _get_possible_block_names(new_block)
		if len(possible_names) != 0:
			var index = rng.randi_range(0, possible_names.size() - 1)
			new_block = possible_names[index]
			grid[pos.z][pos.x].add(new_block, children[new_block], steps[new_block], all_restrictions)
			_new_node(blocks[new_block], str(i+1))
			_add_restrictions(new_block)
			_update_pos(new_block)
		else:
			_new_node(terminal_block, "End")
			complete = false
			break
	if complete == true:
		_new_node(terminal_block, "End")

func _new_node(block_scene, name: String) -> void:
	"""
	The method add a child node of the given block
	to the generator node and sets its name.
	Input: block_scene - scene,
		   name - string.
	"""
	var new_node = block_scene.instance()
	new_node.set_name(name)
	new_node.connect("body_entered", get_node("../Player"), "_on_Road_body_entered")
	new_node.translate(pos * size_coeff)
	add_child(new_node)

	
	
func _add_restrictions(block_id) -> void:
	"""
	The method sets restricted types of blocks
	for the current cell neighbour cells in accordance with
	the current cell block type.
	Input: block_id - int.
	"""
	var up_pos: Vector3 = pos + Vector3(0,0,-1)
	var down_pos: Vector3 = pos + Vector3(0,0,1)
	var left_pos: Vector3 = pos + Vector3(-1,0,0)
	var right_pos: Vector3 = pos + Vector3(1,0,0)
	
	match block_id:
		L1, R2:
			grid[up_pos.z][up_pos.x].restrict(up_restrictions)
			grid[right_pos.z][right_pos.x].restrict(right_restrictions) 
		L2, R1:
			grid[up_pos.z][up_pos.x].restrict(up_restrictions)
			grid[left_pos.z][left_pos.x].restrict(left_restrictions) 
		L3, R4:
			grid[down_pos.z][down_pos.x].restrict(down_restrictions) 
			grid[left_pos.z][left_pos.x].restrict(left_restrictions)
		L4, R3:
			grid[down_pos.z][down_pos.x].restrict(down_restrictions)
			grid[right_pos.z][right_pos.x].restrict(right_restrictions) 
		S1, S3:
			grid[left_pos.z][left_pos.x].restrict(left_restrictions)
			grid[right_pos.z][right_pos.x].restrict(right_restrictions) 
		S2, S4:
			grid[up_pos.z][up_pos.x].restrict(up_restrictions)
			grid[down_pos.z][down_pos.x].restrict(down_restrictions) 


func _update_pos(block_id) -> void:
	"""
	The method shifts position of a cursor in the grid block 
	in accordance with the current cell block type.
	Input: block_id - int.
	"""
	pos += steps[block_id]


func _get_possible_block_names(block_id) -> Array:
	"""
	The method returns array of names of possible children for
	the current cell block type.
	Input: block_id - int.
	Output: possible_names - Array.
	"""
	var possible_names: Array
	for name in children[block_id]:
		if not (name in grid[pos.z][pos.x].restricted):
			possible_names.append(name)
	return possible_names


func _init_grid() -> void:
	"""
	The method fills the grid world with the Cell class instances.
	"""
#	grid.clear()
	for z in range(grid_size):
		var row: Array
		for x in range(grid_size):
			row.append(Cell.new())
		grid.append(row)


func _on_Collision_labels_reset():
	"""
	The method is called then reset button is pressed.
	The method deletes all children of the road generator node 
	and build the new road.
	"""
	for child in get_children():
		child.free()

	for z in range(len(grid)):
		for x in range(len(grid[z])):
			grid[z][x].free()
	grid.clear()
	_build()

