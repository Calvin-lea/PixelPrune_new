from setuptools import setup, find_packages

setup(
    name="pixelprune",
    version="1.0.0",
    description="PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding",
    author="Nan Wang",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
    ],
    entry_points={
        "vllm.general_plugins": [
            "pixelprune = pixelprune.patches.vllm_bootstrap:maybe_apply_patches",
        ],
    },
)
