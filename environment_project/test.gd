extends Node


# Declare member variables here. Examples:
# var a = 2
# var b = "text"

var a = [1,2,3]

# Called when the node enters the scene tree for the first time.
func _ready():
	changer(a)
	print(a)

func changer(x):
	x.append(1)

# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass
