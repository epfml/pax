from setuptools import setup

setup(
    name="paxlib",
    version="0.2",
    description="JAX-like API on top of PyTorch",
    url="http://github.com/epfml/pax",
    author="Thijs Vogels",
    author_email="thijs.vogels@epfl.ch",
    license="MIT",
    packages=["pax"],
    zip_safe=False,
    install_requires=[
          "torch",
          "jax"
    ]
)
