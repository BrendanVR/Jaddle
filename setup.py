from setuptools import setup, find_packages

setup(
    name="jaddle",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "optax",
        "numpy",
        "scipy",
        "highspy",
    ],
    author="Brendan",
    description="A JAX-based saddle solver for linear and convex programs",
    license="MIT",
)
