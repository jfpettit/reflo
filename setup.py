import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name='reflo',
	version='0.01',
	author='Jacob Pettit',
	author_email='jfpettit@gmail.com',
    short_description='TensorFlow implementations of some reinforcement learning algorithms. Intended to be user friendly.',
    long_description=long_description,
	url='https://github.com/jfpettit/rlpack-tf',
	install_requires=['numpy', 'tensorflow==2.7.2', 'gym', 'scipy', 'dm-sonnet',
		'tensorflow-probability', 'pandas', 'matplotlib', 'roboschool', 'mpi4py', 'pybullet']
	,
)
