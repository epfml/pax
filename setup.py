from setuptools import setup, find_packages

setup(
    name="paxlib",
    version="0.3.5",
    description="JAX-like API on top of PyTorch",
    url="http://github.com/epfml/pax",
    author="Thijs Vogels",
    author_email="thijs.vogels@epfl.ch",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
          "torch",
          "jax",
          "jaxlib"
    ]
)
