"""
Setup script for Depth Anything 3.

Includes a post-install hook to optionally install gsplat (requires torch to be
present), and mirrors metadata from pyproject.toml for backward compatibility
with tools that still rely on setup.py.
"""
import pathlib
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        install_gsplat()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        install_gsplat()


def install_gsplat():
    """Install gsplat after torch is available."""
    print("\n" + "=" * 60)
    print("Installing gsplat (requires torch)...")
    print("=" * 60)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"
        ])
        print("✓ gsplat installed successfully")
    except subprocess.CalledProcessError:
        print("⚠ Warning: gsplat installation failed (optional dependency)")
        print("  You can install it manually later with:")
        print("  pip install 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70'")
    print("=" * 60 + "\n")


setup(
    name="depth-anything-3",
    version="0.1.0",
    description="Depth Anything 3 - optimized fork with macOS/MPS & CUDA performance tweaks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Aedelon (Depth Anything 3 optimized fork)",
    license="Apache-2.0",
    url="https://github.com/Aedelon/Depth-Anything-3",
    python_requires=">=3.10,<=3.13",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "pre-commit",
        "trimesh",
        "torch>=2",
        "torchvision",
        "einops",
        "huggingface_hub",
        "imageio",
        "numpy<2",
        "opencv-python",
        "open3d",
        "fastapi",
        "uvicorn",
        "requests",
        "typer>=0.9.0",
        "pillow",
        "omegaconf",
        "evo",
        "e3nn",
        "moviepy==1.0.3",
        "plyfile",
        "pillow_heif",
        "safetensors",
        "pycolmap",
        "xformers; platform_system != 'Darwin'",
    ],
    extras_require={
        "app": ["gradio>=5", "pillow>=9.0"],
        "optimizations": ["xformers; platform_system != 'Darwin'"],
        "all": ["depth-anything-3[app]"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "da3=depth_anything_3.cli:app",
        ],
    },
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
