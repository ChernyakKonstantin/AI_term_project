Входные данные - карта глубины со стереокамеры с передней части автомобиля размером 64x64 пикселей.
Награды:
-движение вперед: 1.0
-движение назад: -1.5 (теперь -50.0)
-движение вперед и вправо: 1.0 
-движение вперед и влево: 1.0
-движение назад и вправо: -1.5 (теперь -50.0)
-движение назад и влево: -1.5 (теперь -50.0)
-Столкновение: -100.

Столкновение и, следовательно, завершение эпизода определяется сравнением последовательных кадров. Если кадры одинаковые - значит произошло столкновение.
После непродолжительного обучения выявлено следующее:
-Автомобиль не едет дальше первого поворота. Движется вперед-назад.
-Функция потерь существенно уменьшается, однако объем данных малый, возможно переобучение.

Дальнейший план:
-Добавить стереокамеру на заднюю часть автомобиля.
-Существенно увеличить штраф за движение назад, чтобы стимулировать агента избегать этих действий. (Done)
-Поэксперементировать с парметрами. (Done)
-Добавить RNN/LSTM.
