cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ratSS)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/ratSS")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Duoyao LIANG (CarMeN)")
set(EXTENSION_DESCRIPTION "Rat Brain Stroke semgentation")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/ratSS.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/ratSS/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ratbrainseg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
