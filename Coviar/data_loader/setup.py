from distutils.core import setup, Extension
import numpy as np

coviar_utils_module = Extension('coviar',
		sources = ['coviar_data_loader.c'],
		include_dirs=[np.get_include(), '/home/mmstu_a/lxq/ffmpeg/include/'],
		extra_compile_args=['-DNDEBUG', '-O3','-std=c99'],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale', '-L/home/mmstu_a/lxq/ffmpeg/lib/']
)

setup ( name = 'coviar',
	version = '0.1',
	description = 'Utils for coviar training.',
	ext_modules = [ coviar_utils_module ]
)
