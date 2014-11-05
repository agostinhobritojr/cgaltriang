TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
QMAKE_CXXFLAGS += -std=c++0x -frounding-math
SOURCES += main.cpp

LIBS+= -lCGAL -lgmp
