import PySimpleGUI as sg
import random
from PIL import Image, ImageDraw
import numpy as np
import os
from matplotlib import image as mpim
from matplotlib import pyplot as plt
import microstructpy as msp

# matrix is as such matrixOld[position up and down][position left and right]

token_image = [0]


def paint_all(new_image_size):
	pixel_array = [[0] * rows for i in range(columns)]

	for i in range(rows):
		for j in range(columns):
			pixel_array[i][j] = colors[matrixOld[i][j]]

	array = np.array(pixel_array, dtype=np.uint8)
	new_image = Image.fromarray(array)

	if stretch_image:
		new_image = new_image.resize((new_image_size, new_image_size), Image.NEAREST)
		# not that great resizing option, can be disabled and quality will stay default but size will be shrunk

	new_image.save('Microstructure.png')

	# file_in = "Microstructure.png"
	# img = Image.open(file_in)
	#
	# file_out = "test1.bmp"
	# img.save(file_out)

	window['-MICROSTRUCTURE-'].update(r'C:\Users\Acer\PycharmProjects\Microstructure\Microstructure.png')


# INITIALIZATION
image_size = 500
# for max size 500 there are no problems xd
MAX_SIZE = 500  # can be anything
MAX_SEEDING_ROUNDS = 100  # can be anything
MAX_ROUNDING_ROUNDS = 20
loops = 0
rows = columns = 250
matrixOld = [[0] * columns for i in range(rows)]
matrixNew = [[0] * columns for i in range(rows)]
colors = [
	(255, 255, 255),  # white for 0 - empty cell
	(255, 0, 0), (192, 0, 0), (128, 0, 0), (64, 0, 0),  # red hues id 1, 2, 3, 4
	(0, 255, 0), (0, 192, 0), (0, 128, 0), (0, 64, 0),  # green hues id 5, 6, 7, 8
	(0, 0, 255), (0, 0, 192), (0, 0, 128), (0, 0, 64),  # blue hues id 9, 10, 11, 12
	(255, 255, 0), (192, 192, 0), (128, 128, 0), (64, 64, 0),  # yellow hues 13, 14, 15, 16
	(0, 0, 0)]  # black for border
#  colors = [(255, 255, 255), (33, 146, 255), (56, 229, 77), (156, 255, 46), (253, 255, 0)]
stretch_image = False
boundaries_marked = False
max_number_of_seeds = len(colors)-2

# describes how the gui window looks like
layout = [
	[sg.Text('Matrix size:'), sg.Text(size=(30, 1), key='-OUTPUT-')],
	[sg.Text('Size:'), sg.Input("200", key='-Size-', size=4),
		sg.Text('Seed types:'), sg.Input("16", key='-Seeds-', size=4),
		sg.Text('Seeding rounds:'), sg.Input("15", key='-Seeding rounds-', size=4),
		sg.Text('Rounding rounds:'), sg.Input("3", key='-Rounding rounds-', size=4)],
	[sg.Button('Clear completely'), sg.Button('Set max'), sg.Button('Set custom'), sg.Button('Stretch image to:'),
		sg.Input("500", key='-Image size-', size=4)],
	[sg.Button('Spread once VN'), sg.Button('Spread once RP'),
		sg.Button('Spread completely'), sg.Button('Round grains')],
	[sg.Button('Mark boundaries'), sg.Button('Clear colours'), sg.Button("Generate mesh and input file")],  # ButtonMenu
	[sg.Image(r'C:\Users\Acer\PycharmProjects\Microstructure\Empty.png', key='-MICROSTRUCTURE-')]]

window = sg.Window('Microstructure CA/MC', layout)

while True:  # Event Loop
	event, values = window.read()

	if event == sg.WIN_CLOSED or event == 'Exit':
		break

	if event == 'Generate mesh and input file':
		print("Generating mesh and abaqus input file")
		# Read in image
		image_basename = 'Microstructure.png'
		image_path = os.path.dirname(__file__)
		image_filename = os.path.join(image_path, image_basename)
		image = mpim.imread(image_filename)
		im_brightness = image[:, :, 0]

		# Bin the pixels
		br_bins = [0.00, 0.50, 1.00]

		bin_nums = np.zeros_like(im_brightness, dtype='int')
		for i in range(len(br_bins) - 1):
			lb = br_bins[i]
			ub = br_bins[i + 1]
			mask = np.logical_and(im_brightness >= lb, im_brightness <= ub)
			bin_nums[mask] = i

		# Define the phases
		phases = [{'color': c, 'material_type': 'amorphous'} for c in ('C0', 'C1')]

		# Create the polygon mesh
		m, n = bin_nums.shape
		x = np.arange(n + 1).astype('float')
		y = m + 1 - np.arange(m + 1).astype('float')
		xx, yy = np.meshgrid(x, y)
		pts = np.array([xx.flatten(), yy.flatten()]).T
		kps = np.arange(len(pts)).reshape(xx.shape)

		n_facets = 2 * (m + m * n + n)
		n_regions = m * n
		facets = np.full((n_facets, 2), -1)
		regions = np.full((n_regions, 4), 0)
		region_phases = np.full(n_regions, 0)

		facet_top = np.full((m, n), -1, dtype='int')
		facet_bottom = np.full((m, n), -1, dtype='int')
		facet_left = np.full((m, n), -1, dtype='int')
		facet_right = np.full((m, n), -1, dtype='int')

		k_facets = 0
		k_regions = 0
		for i in range(m):
			for j in range(n):
				kp_top_left = kps[i, j]
				kp_bottom_left = kps[i + 1, j]
				kp_top_right = kps[i, j + 1]
				kp_bottom_right = kps[i + 1, j + 1]

				# left facet
				if facet_left[i, j] < 0:
					fnum_left = k_facets
					facets[fnum_left] = (kp_top_left, kp_bottom_left)
					k_facets += 1

					if j > 0:
						facet_right[i, j - 1] = fnum_left
				else:
					fnum_left = facet_left[i, j]

				# right facet
				if facet_right[i, j] < 0:
					fnum_right = k_facets
					facets[fnum_right] = (kp_top_right, kp_bottom_right)
					k_facets += 1

					if j + 1 < n:
						facet_left[i, j + 1] = fnum_right
				else:
					fnum_right = facet_right[i, j]

				# top facet
				if facet_top[i, j] < 0:
					fnum_top = k_facets
					facets[fnum_top] = (kp_top_left, kp_top_right)
					k_facets += 1

					if i > 0:
						facet_bottom[i - 1, j] = fnum_top
				else:
					fnum_top = facet_top[i, j]

				# bottom facet
				if facet_bottom[i, j] < 0:
					fnum_bottom = k_facets
					facets[fnum_bottom] = (kp_bottom_left, kp_bottom_right)
					k_facets += 1

					if i + 1 < m:
						facet_top[i + 1, j] = fnum_bottom
				else:
					fnum_bottom = facet_bottom[i, j]

				# region
				region = (fnum_top, fnum_left, fnum_bottom, fnum_right)
				regions[k_regions] = region
				region_phases[k_regions] = bin_nums[i, j]
				k_regions += 1

		print("Generating polygon mesh")
		pmesh = msp.meshing.PolyMesh(pts, facets, regions, seed_numbers=range(n_regions), phase_numbers=region_phases)

		# Create the triangle mesh
		print("Generating triangle mesh")
		tmesh = msp.meshing.TriMesh.from_polymesh(pmesh, phases=phases, min_angle=20)

		# Plot triangle mesh
		print("Plotting triangle mesh")
		fig = plt.figure()
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		fig.add_axes(ax)

		fcs = [phases[region_phases[r]]['color'] for r in tmesh.element_attributes]
		tmesh.plot(facecolors=fcs, edgecolors='k', lw=0.2)

		plt.axis('square')
		plt.xlim(x.min(), x.max())
		plt.ylim(y.min(), y.max())
		plt.axis('off')

		# Save plot and copy input file
		print("Saving plot")
		plot_basename = 'Generated/trimesh.png'
		file_dir = os.path.dirname(os.path.realpath(__file__))
		filename = os.path.join(file_dir, plot_basename)
		dirs = os.path.dirname(filename)
		if not os.path.exists(dirs):
			os.makedirs(dirs)
		plt.savefig(filename, bbox_inches='tight', pad_inches=0)

		print("Saving input file")
		tmesh.write(r'Generated/abaqus_input.inp', 'abaqus', None, pmesh)

		print("Post processing input file")
		with open("Generated/abaqus_input.inp", "r") as input_file:
			with open("temp.inp", "w") as output_file:
				# iterate all lines from file
				for line in input_file:
					# if line starts with substring 'time' then don't write it in temp file
					if not line.strip("\n").startswith('Surface-') and not line.strip("\n").startswith('*Surface, name=Ext-'):
						output_file.write(line)

		# replace file with original name
		os.replace('temp.inp', 'Generated/processed_abaqus_input.inp')

		print("Generated mesh and abaqus input file successfully")
		window['-MICROSTRUCTURE-'].update(r'C:\Users\Acer\PycharmProjects\Microstructure\Generated\trimesh.png')

	if event == 'Clear completely':
		matrixOld = [[0] * columns for i in range(rows)]
		matrixNew = [[0] * columns for i in range(rows)]
		paint_all(image_size)

	if event == 'Clear colours':
		for i in range(rows):
			for j in range(columns):
				if matrixOld[i][j] != max_number_of_seeds + 1:
					matrixOld[i][j] = 0
		paint_all(image_size)
		print("Colours cleared")

	if event == 'Mark boundaries':
		if not boundaries_marked:
			for i in range(rows):
				for j in range(columns):
					# instead of what you have now you can go to the right and down,
					# and check both up and left at the same time so that already marked cells
					# won't be involved in further checks

					# vertical walk through every row
					# if i == columns - 1:
					# 	next_cell_id = - 1  # doesn't exist
					# else:
					if i != columns - 1:
						next_cell_id = matrixOld[i + 1][j]
						current_cell_id = matrixOld[i][j]

						# mark the top one
						if next_cell_id != current_cell_id:
							matrixOld[i][j] = max_number_of_seeds + 1  # last color is black, used only for boundaries

					# horizontal walk through every row
					if j != rows - 1:
						next_cell_id = matrixOld[i][j + 1]
						current_cell_id = matrixOld[i][j]

						# mark the left one
						if next_cell_id != current_cell_id:
							matrixOld[i][j] = max_number_of_seeds + 1

			boundaries_marked = True
			print("Boundaries marked")
		paint_all(image_size)

	if event == 'Stretch image to:':
		image_size = int(values['-Image size-'])
		stretch_image = True
		paint_all(image_size)

	if event == 'Round grains':
		rounding_rounds = int(values['-Rounding rounds-'])
		upper_limit = columns
		print("Rounding rounds: ")
		for i in range(rounding_rounds):
			print("\b", end='')
			print(i+1, end='')
			# choosing random cel to lower its ENERGY if possible (MC - Monte Carlo part)
			# which is to check all it's neighbours and change it int the type which most of the neighbours are

			# array for keeping track of which cells have been affected
			# 0 - not affected 1, or more affected this many times
			affected_cells_array = [[0] * columns for i in range(rows)]

			all_affected = False
			cells_affected_counter = 0

			while not all_affected:
				# Completely random cell to take into consideration
				x = round(random.random() * upper_limit - 0.5)
				y = round(random.random() * upper_limit - 0.5)

				if cells_affected_counter == columns * rows:
					all_affected = True

				# affect each only once
				if affected_cells_array[y][x] == 0:
					# register it being affected one more time
					affected_cells_array[y][x] += 1

					cells_affected_counter += 1

					# ensuring the code doesn't reach out of the matrix (would cause errors)
					if x == -1:
						x = 0
					if x == columns - 1:
						x = columns - 2
					if y == -1:
						y = 0
					if y == rows - 1:
						y = rows - 2
					# now we have random cell position (x,y) (we don't need its value)
					# now we have to check its neighbours, and to get outside of the matrix (-1,-1) (250,250)=(rows,cols)
					# pattern is as such:

					# [y - 1][x - 1] 	[y - 1][x] 	[y - 1][x + 1]
					# [y][x - 1]		chosen y,x	[y][x + 1]
					# [y + 1][x - 1] 	[y + 1][x] 	[y + 1][x + 1]
					neighbouring_ids = [
						matrixOld[y - 1][x - 1],
						matrixOld[y][x - 1],
						matrixOld[y + 1][x - 1],
						matrixOld[y - 1][x],
						matrixOld[y + 1][x],
						matrixOld[y - 1][x + 1],
						matrixOld[y][x + 1],
						matrixOld[y + 1][x + 1]
					]
					unique, counts = np.unique(neighbouring_ids, return_counts=True)
					highest_cell_id_occurrence = max(counts)
					its_index = list(counts).index(highest_cell_id_occurrence)
					according_cell_id = list(unique)[its_index]
					# example how the code above works
					# 1 1 2
					# 5 ? 2
					# 5 5 5
					# unique = [1 2 5]
					# counts = [2 2 4]
					# highest_cell_id_occurrence = 4
					# its_index = 2
					# according_cell_id = 5
					# and now that we have the new id value for our cell to receive the last step is to assign it
					matrixOld[y][x] = according_cell_id
		paint_all(image_size)
		print("\nRounding complete")

	if event == 'Set max':
		boundaries_marked = False
		columns = rows = MAX_SIZE
		matrixOld = [[0] * columns for i in range(rows)]
		matrixNew = [[0] * columns for i in range(rows)]
		number_of_seeds = max_number_of_seeds
		seeding_rounds = MAX_SEEDING_ROUNDS
		rounding_rounds = MAX_ROUNDING_ROUNDS
		msg1 = str(columns) + "x" + str(rows) + " and: " + str(seeding_rounds) + " rounds of: "
		msg2 = str(number_of_seeds) + " seeds."
		window['-OUTPUT-'].update(msg1 + msg2)

		for n in range(seeding_rounds):
			seed_id = 1
			for i in range(number_of_seeds):
				matrixOld[round(random.random() * rows) - 1][round(random.random() * columns) - 1] = seed_id
				seed_id += 1

		paint_all(image_size)

	if event == 'Set custom':
		boundaries_marked = False

		columns = int(values['-Size-'])
		if columns > MAX_SIZE:
			columns = MAX_SIZE
		rows = columns

		number_of_seeds = int(values["-Seeds-"])
		if number_of_seeds > max_number_of_seeds:
			number_of_seeds = max_number_of_seeds

		seeding_rounds = int(values["-Seeding rounds-"])
		if seeding_rounds > MAX_SEEDING_ROUNDS:
			seeding_rounds = MAX_SEEDING_ROUNDS

		msg1 = str(columns) + "x" + str(rows) + " and: " + str(seeding_rounds) + " rounds of: "
		msg2 = str(number_of_seeds) + " seeds."
		window['-OUTPUT-'].update(msg1 + msg2)

		# matrixOld = [[0] * columns for i in range(rows)]
		# matrixNew = [[0] * columns for i in range(rows)]

		for n in range(seeding_rounds):
			seed_id = 1
			for i in range(number_of_seeds):
				matrixOld[round(random.random() * rows) - 1][round(random.random() * columns) - 1] = seed_id
				seed_id += 1

		paint_all(image_size)

	if event == 'Spread once VN':
		# cell pattern
		# 1 2 3
		# 4 ? 5
		# 6 7 8
		# VN - Von Neumann is 2->5->7->4

		for i in range(rows):
			for j in range(columns):

				# rewrite previously filled cells
				if matrixOld[i][j] != 0:
					matrixNew[i][j] = matrixOld[i][j]

				if matrixOld[i][j] == 0:
					if i < rows - 1 and matrixOld[i + 1][j] != 0:
						# fill current cell as below
						# go UP
						s = matrixOld[i + 1][j]
						matrixNew[i][j] = s

					if j > 0 and matrixOld[i][j - 1] != 0:
						# fill current cell as the one to the left
						# go RIGHT
						s = matrixOld[i][j - 1]
						matrixNew[i][j] = s

					if i > 0 and matrixOld[i - 1][j] != 0:
						# fill current cell as above
						# go DOWN
						s = matrixOld[i - 1][j]
						matrixNew[i][j] = s

					if j < columns - 1 and matrixOld[i][j + 1] != 0:
						# fill current cell as the one to the right
						# go LEFT
						s = matrixOld[i][j + 1]
						matrixNew[i][j] = s

		paint_all(image_size)

		# New to old
		for i in range(rows):
			for j in range(columns):
				matrixOld[i][j] = matrixNew[i][j]

	if event == 'Spread once RP':

		# random pentagonal spread
		# pattern is as such:
		# [y - 1][x - 1] 	[y - 1][x] 	[y - 1][x + 1]
		# [y][x - 1]		chosen x,y	[y][x + 1]
		# [y + 1][x - 1] 	[y + 1][x] 	[y + 1][x + 1]
		# 0 1 2
		# 7 # 3
		# 6 5 4

		pn = round(random.random() * (3 + 1) - 0.5)
		print("Pattern: " + str(pn))
		# pattern_number

		for i in range(rows):
			for j in range(columns):

				# rewrite previously filled cells
				if matrixOld[i][j] != 0:
					matrixNew[i][j] = matrixOld[i][j]

				# neighbour_choice_order
				nco = [
					[[i, j - 1], [i - 1, j - 1], [i - 1, j], [i - 1, j + 1], [i, j + 1]],  # 0 up
					[[i - 1, j], [i - 1, j + 1], [i, j + 1], [i + 1, j + 1], [i + 1, j]],  # 1 right
					[[i, j + 1], [i + 1, j + 1], [i + 1, j], [i + 1, j - 1], [i, j - 1]],  # 2 down
					[[i + 1, j], [i + 1, j - 1], [i, j - 1], [i - 1, j - 1], [i - 1, j]],  # 3 left
				]
				if 0 < i < rows - 1 and 0 < j < columns - 1:
					if matrixOld[i][j] != 0:
						for k in range(5):  # 5 iterations 0-4
							# fill only if empty
							if matrixNew[nco[pn][k][0]][nco[pn][k][1]] == 0:
								matrixNew[nco[pn][k][0]][nco[pn][k][1]] = matrixOld[i][j]

		paint_all(image_size)
		# New to old
		for i in range(rows):
			for j in range(columns):
				matrixOld[i][j] = matrixNew[i][j]

	if event == 'Spread completely':  # Untested
		loops = 0
		complete = False
		VN_turn = True
		print("Spreading rounds: ")
		while not complete:
			complete = True  # assume it's complete from the beginning unless changed later

			loops += 1
			print("\b\b", end='')
			print(loops, end='')

			pn = round(random.random() * (3 + 1) - 0.5)
			# pattern_number (of direction of random pentagonal growth)

			# so its one Von Neumann and one Random Pentagonal, alternating
			VN_turn = not VN_turn

			for i in range(rows):
				for j in range(columns):

					# rewrite previously filled cells
					if matrixOld[i][j] != 0:
						matrixNew[i][j] = matrixOld[i][j]
					else:  # There are still empty cells to fill
						complete = False  # not complete bc there are still empty cells

					if VN_turn:
						if matrixOld[i][j] == 0:
							if i < rows - 1 and matrixOld[i + 1][j] != 0:
								# fill current cell as below
								# go UP
								s = matrixOld[i + 1][j]
								matrixNew[i][j] = s

							if j > 0 and matrixOld[i][j - 1] != 0:
								# fill current cell as the one to the left
								# go RIGHT
								s = matrixOld[i][j - 1]
								matrixNew[i][j] = s

							if i > 0 and matrixOld[i - 1][j] != 0:
								# fill current cell as above
								# go DOWN
								s = matrixOld[i - 1][j]
								matrixNew[i][j] = s

							if j < columns - 1 and matrixOld[i][j + 1] != 0:
								# fill current cell as the one to the right
								# go LEFT
								s = matrixOld[i][j + 1]
								matrixNew[i][j] = s
					else:  # (if RP_turn) it's designed so that every single filled cell evokes RP growth (works but slow)
						nco = [
							[[i, j - 1], [i - 1, j - 1], [i - 1, j], [i - 1, j + 1], [i, j + 1]],  # 0 up
							[[i - 1, j], [i - 1, j + 1], [i, j + 1], [i + 1, j + 1], [i + 1, j]],  # 1 right
							[[i, j + 1], [i + 1, j + 1], [i + 1, j], [i + 1, j - 1], [i, j - 1]],  # 2 down
							[[i + 1, j], [i + 1, j - 1], [i, j - 1], [i - 1, j - 1], [i - 1, j]],  # 3 left
						]
						if 0 < i < rows - 1 and 0 < j < columns - 1:
							if matrixOld[i][j] != 0:
								for k in range(5):  # 5 iterations 0-4
									# fill only if empty
									if matrixNew[nco[pn][k][0]][nco[pn][k][1]] == 0:
										matrixNew[nco[pn][k][0]][nco[pn][k][1]] = matrixOld[i][j]
			# New to old
			for i in range(rows):
				for j in range(columns):
					matrixOld[i][j] = matrixNew[i][j]
		paint_all(image_size)
		print("\nSpreading complete")

window.close()
