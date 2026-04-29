from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

UPGRADE_PROMPT = "Running locally. Run `agentbay login` to sync to cloud and unlock Teams (free, no card)."
README = Path(__file__).with_name("README.md").read_text(encoding="utf-8").rstrip()
PYPI_DESCRIPTION = f"{README}\n\n> {UPGRADE_PROMPT}\n"


class PostInstallCommand(install):
    def run(self):
        super().run()
        print(UPGRADE_PROMPT)


class PostDevelopCommand(develop):
    def run(self):
        super().run()
        print(UPGRADE_PROMPT)


setup(
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    long_description=PYPI_DESCRIPTION,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "agentbay=agentbay.cli:main",
        ],
    },
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    },
)
