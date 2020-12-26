from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from utils import *
import sys
from numpy import cos
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pdb
from QtImageViewer import PhotoViewer

class CustomDialog(QDialog):
	def __init__(self, *args, **kwargs):
		super(CustomDialog, self).__init__(*args, **kwargs)

		self.setWindowTitle("Choose Folder")
		
		self.directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

class ErrorDialog(QDialog):
	def __init__(self, *args, **kwargs):
		super(ErrorDialog, self).__init__(*args, **kwargs)

		self.setWindowTitle("Error")
		label = QLabel("ERROR CHOOSE AGAIN")
		label.setAlignment(Qt.AlignCenter)
		self.layout = QVBoxLayout()
		self.layout.addWidget(label)
		self.setLayout(self.layout)

class Visualization(HasTraits):

	scene = Instance(MlabSceneModel, ())
	@on_trait_change('scene.activated')
	def set_plot(self, pc, bbox, classes):	
		self.scene.background = (0,0,0)
		mlab.clf(figure = self.scene.mayavi_scene)
		mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=(0,1,0), mode='point', colormap = 'gnuplot', scale_factor=0.4, figure = self.scene.mayavi_scene)
		print(type(pc))
		num = len(bbox)
		for n in range(num):
			b = bbox[n]
			for k in range(0,4):
			#http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
				i,j=k,(k+1)%4
				self.scene.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=(0,0,1), tube_radius=None, line_width=2, figure = self.scene.mayavi_scene)

				i,j=k+4,(k+1)%4 + 4
				self.scene.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=(0,0,1), tube_radius=None, line_width=2, figure = self.scene.mayavi_scene)

				i,j=k,k+4
				self.scene.mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=(0,0,1), tube_radius=None, line_width=2, figure = self.scene.mayavi_scene)
			coord = np.max(b, axis = 0)
			mlab.text3d(coord[0], coord[1], coord[2], classes[n], color=(1,0,0), scale=1.0, figure=self.scene.mayavi_scene)
	view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
					 height=250, width=300, show_label=False),
				resizable=True )

class MayaviQWidget(QWidget):
	def __init__(self, parent, pc = None, bbox = None, classes = None):
		QWidget.__init__(self, None)
		layout = QVBoxLayout(self)
		layout.setContentsMargins(0,0,0,0)
		layout.setSpacing(0)
		self.visualization = Visualization()
		self.visualization.set_plot(pc, bbox, classes)

		self.ui = self.visualization.edit_traits(parent=self,
												 kind='subpanel').control
		layout.addWidget(self.ui)
		self.ui.setParent(self)

class MainWindow(QMainWindow):
	def __init__(self, parent = None):
		QMainWindow.__init__(self, parent)

		self.main_folder = 'training'
		self.generator = Generator(self.main_folder)
		self.setWindowTitle("Simple Image viewer")
		
		self.main_layout = QVBoxLayout()
		self.sublayout = QHBoxLayout()

		self.viewer1 = PhotoViewer(self)
		self.viewer1.setPhoto(QPixmap('temp.png'))
		self.viewer1.toggleDragMode()
		

		
		self.viewer2 = PhotoViewer(self)
		self.viewer2.setPhoto(QPixmap('temp1.png'))
		self.viewer2.toggleDragMode()
		
		
		
		self.widget = QWidget()
		container = QWidget()
		self.mayavi_widget = MayaviQWidget(container, self.generator.pc, self.generator.bbox, self.generator.classes)

		
		self.sublayout.addWidget(self.viewer1)
		self.sublayout.addWidget(self.viewer2)

		self.button_layout = self.set_base_layout()
		self.main_layout.addWidget(self.mayavi_widget)
		self.main_layout.addLayout(self.sublayout)
		self.main_layout.addLayout(self.button_layout)
		self.main_folder = None

		
		self.widget.setLayout(self.main_layout)
		self.setCentralWidget(self.widget)



	def set_base_layout(self):

		
		layout = QGridLayout()

		btn = QPushButton('Folder')
		layout.addWidget(btn, 0,0)
		btn.pressed.connect(lambda :self.open_dialog())


		btn = QPushButton('Next')
		layout.addWidget(btn, 0,1)
		btn.pressed.connect(lambda :self.change_image('Next'))
		btn = QPushButton('Previous')
		layout.addWidget(btn, 0,2)
		btn.pressed.connect(lambda :self.change_image('Previous'))

		return layout

	def change_image(self, s):
		if s == 'Next':
			img = self.generator.getNextImage()
			if img == 1:
				self.viewer1.setPhoto(QPixmap('temp.png'))
				self.viewer2.setPhoto(QPixmap('temp1.png'))
				self.mayavi_widget.visualization.set_plot(self.generator.pc, self.generator.bbox, self.generator.classes)
		else:
			img = self.generator.getPreviousImage()
			if img == 1:
				self.viewer1.setPhoto(QPixmap('temp.png'))
				self.viewer2.setPhoto(QPixmap('temp1.png'))
				self.mayavi_widget.visualization.set_plot(self.generator.pc, self.generator.bbox, self.generator.classes)




	def open_dialog(self):
		dlg = CustomDialog(self)
		if len(dlg.directory) != 0:
			self.generator = Generator(dlg.directory)
		elif len(dlg.directory) == 0 and self.main_folder == None:
			err = ErrorDialog()
			err.exec_()
		else:
			pass


def main():

	app = QApplication([])
	window = MainWindow()
	window.show()

	app.exec_()

if __name__ ==  "__main__":
	main()


