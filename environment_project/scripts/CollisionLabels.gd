extends Control

signal reset



func _on_Player_collision():
	"""
	The method is called if the car collides a wall.
	The method shows a collision message and blocks input.
	"""
	show()	

	
func _physics_process(_delta):
	"""
	The method is called every _delta seconds.
	The method checks if reset button is pressed.
	If true, it hides a collision message and raise a sygnal
	for player to reset and unblock input.
	"""
	if Input.is_action_pressed("reset"):
		hide()
		emit_signal("reset")
		
