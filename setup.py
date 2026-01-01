from setuptools import setup, find_packages

setup(
    name="AntiSPencoder",           # pip install 时用的名字
    version="1.0.0",
    author="Rui Gan",
    description="CDR3 and antigen peptide sequence embeddings",
    packages=find_packages(),    # 自动查找所有包
    include_package_data=True,

)

