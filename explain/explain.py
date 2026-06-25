# -*- coding: utf-8 -*-
# feon-heat-transfer
#
# A heat conduction and convection analysis extension module based on the Feon
# finite element framework.
#
# Acknowledgment: This module is based on the Feon open-source FEM framework
# developed by YaoYao Pei.
# Feon GitHub: https://github.com/YaoyaoBae/Feon
# Feon Author Email: yaoyao.bae@foxmail.com
#
#
# 1. What is this?
#
# feon-heat-transfer adds heat conduction and convection analysis capabilities
# to the Feon framework. Feon originally supports only structural mechanics (SA)
# and fluid mechanics (FFA). This module enables temperature field analysis.
#
#
# 2. Installation Steps
#
# Step 1: Install Feon
#
# pip install feon
#
# Feon official repository: https://github.com/YaoyaoBae/Feon
#
# Step 2: Replace Feon's base.py (required - the module will not work otherwise)
#
# Feon was originally designed for mechanics, so its nodes only have force and
# displacement attributes. The feon/base.py in this repository is already
# modified. Use it to directly replace the original Feon file.
#
# Locate the Feon installation directory:
# python -c "import feon; print(feon.__path__)"
#
# Copy feon/base.py from this repository and overwrite the original file.
#
# If you prefer not to replace the file, you can manually add the following
# code to the original base.py:
#
# In the NodeBase class, add:
# @property
# def nCk(self):
#     return self._nCk
#
# def set_nCk(self,val):
#     self._nCk = val
#
# def get_nCk(self):
#     return self._nCk
#
# In the SystemBase class, add:
# @property
# def nCk(self):
#     return self._nCk
#
# Step 3: Place the tha folder into the Feon directory
#
# Copy the feon/tha/ folder from this repository into the Feon installation
# directory, so that tha sits alongside sa and ffa (same level):
#
# site-packages/feon/
# ©À©¤©¤ sa/              # Structural analysis (Feon built-in)
# ©À©¤©¤ ffa/             # Fluid analysis (Feon built-in)
# ©À©¤©¤ tha/             # Heat transfer analysis (this module, manually added)
# ©¦   ©À©¤©¤ __init__.py
# ©¦   ©À©¤©¤ node.py
# ©¦   ©À©¤©¤ element.py
# ©¦   ©À©¤©¤ system.py
# ©¦   ©À©¤©¤ solver.py
# ©¦   ©¸©¤©¤ post_process.py
# ©¸©¤©¤ base.py          # Replaced in Step 2
#
#
# 3. Acknowledgment
#
# This module is based on the Feon open-source finite element framework
# developed by YaoYao Pei.
#
# Citation:
# Pei, Y. (2017). Feon: A Free and Open-Source Finite Element Analysis Python
# Framework. Hubei University of Technology.
#
# Feon GitHub: https://github.com/YaoyaoBae/Feon
# Author Email: yaoyao.bae@foxmail.com
# Feon QQ Group: 555809224
#
#
# 4. License
#
# MIT License
#
# Disclaimer: This module is a third-party extension and is not an official
# Feon project.