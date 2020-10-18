extends KinematicBody
""" 
This class provides the player and logic of his interaction with the environment.
Constant value SPEED defines the car speed in th engine units.
Constant value ANGLE defines the angle steering in radians.
Variable INIT_TRANSFORM defines the car postion at the beginning of an episode.
"""


var vel_vec = Vector3()
var cur_angle

var INIT_TRANSFORM = Transform(Vector3(0, 0, -0.2), Vector3(0, 0.2, 0), Vector3(0.2, 0, 0), Vector3(62, 0.276, 62))

const SPEED = 1
const ANGLE = deg2rad(2)

var block_input: bool = false

signal collision


func _physics_process(_delta):
	"""
	The method is called every _delta seconds.
	The method check if input is enabled and move the car 
	in accordance with the pressed buttons.
	"""
	if block_input == false:
		vel_vec = Vector3(0,0,0)
		cur_angle =  get_rotation().y
		if Input.is_action_pressed("ui_up"):
			vel_vec.x = -SPEED * cos(cur_angle)
			vel_vec.z = SPEED * sin(cur_angle)
		if Input.is_action_pressed("ui_down"):
			vel_vec.x = +SPEED * cos(cur_angle)
			vel_vec.z = -SPEED * sin(cur_angle)
		if Input.is_action_pressed("ui_right"):
			rotate_y(-ANGLE)
		if Input.is_action_pressed("ui_left"):
			rotate_y(ANGLE)
		
		move_and_slide(vel_vec, Vector3.UP)


func _on_Road_body_entered(body):
	"""
	The method is called then the car collides the road walls.
	"""
	if body.name == self.name:
		game_over()

func game_over():
	"""
	The method disables input and raises a signal for she collison message.
	"""
	block_input = true
	emit_signal("collision")

func _on_Collision_labels_reset():
	"""
	The method is called when the reset button is pressed.
	The method enables input and places the car to the initial position.
	"""
	block_input = false
	set_global_transform(INIT_TRANSFORM)


