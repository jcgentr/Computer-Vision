import pyautogui as pag

# mouseNow.py - Display the mouse cursor's current position on the screen
print('Press Ctrl-C to quit.')

try:
	while True:
		x,y = pag.position()
		posStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
		print(posStr, end='')
		print('\b' * len(posStr), end='', flush=True)
except KeyboardInterrupt:
	print('\nDone.')

