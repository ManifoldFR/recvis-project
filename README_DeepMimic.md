# Installation DeepMimic

Hypothèses: on utilise des environnements conda

## Dépendances

J'ai installé tout avec conda dans les environnements où je compile (Eigen, Glut, Glew, pyopengl sont dans les repo conda).
```bash
conda install eigen glut glew pyopengl
```


## Bullet

Pour forcer CMake a regarder dans les dossiers conda (de l'env actif), faire
```bash
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
```
(utiliser CONDA_PREFIX_1 pour forcer l'environnement de base). 
J'ai compilé dans mon environnement de base.

Il est possible qu'il soit nécessaire de modifer le script d'installation `build_cmake_pybullet_double.sh` comme indiqué sur le billet https://github.com/xbpeng/DeepMimic/issues/82


## Makefile modifié pour DeepMimic

J'ai compilé dans mon environnement TensorFlow.
J'ai fait les modifications suivantes:
```makefile
# EIGEN_DIR = ../../libraries/eigen
EIGEN_DIR = $(CONDA_PREFIX)/include/eigen3
# BULLET_INC_DIR = ../../libraries/bullet3
BULLET_INC_DIR = ../../bullet3-2.89/src

# PYTHON_INC = /usr/include/python3.6m
PYTHON_INC = $(CONDA_PREFIX)/include/python3.7m
# PYTHON_LIB = /usr/lib/ -lpython3.6m
PYTHON_LIB = $(CONDA_PREFIX)/lib -lpython3.7m

INC = -I./ \
	-I$(EIGEN_DIR) \
	-I$(BULLET_INC_DIR) \
	-I$(CONDA_PREFIX)/include

LIBS = -lGLEW -lGL -lGLU -lglut -lBulletDynamics -lBulletCollision -lLinearMath -lm -lstdc++ -Wl,-rpath=/usr/local/lib
```
Pour forcer le linker a regarder les libs conda, j'ai fait
```bash
export LIBRARY_PATH=$CONDA_PREFIX
```
(aussi possible d'ajouter `-L$(CONDA_PREFIX)` dans la variable LIBS du Makefile).

### Linker runtime

On a ajouté `-Wl,-rpath=/usr/local/lib` dans les flags passés au compilateur (variable CFLAGS dans le Makefile) pour forcer le runtime à regarder au bon endroit pour Bullet3, dans `/usr/local/lib`. Si ce n'est pas fait il faut faire
```bash
export LD_LIBRARY_PATH=/usr/local/lib
```
avant de lancer DeepMimic.

## Autres écueils

**Si le 1er exemple crashe quand on balance des boîtes** Suivre les instrutions du gars et modifier le code: https://github.com/xbpeng/DeepMimic/issues/58#issuecomment-502038564
