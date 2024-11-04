# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.26.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(rerun_sdk-populate NONE)



include(ExternalProject)
ExternalProject_Add(rerun_sdk-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "TRUE" "URL" "https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip"
                    SOURCE_DIR          "/home/hice1/kshenoy8/scratch/stereosquad/openMVG_Build/_deps/rerun_sdk-src"
                    BINARY_DIR          "/home/hice1/kshenoy8/scratch/stereosquad/openMVG_Build/_deps/rerun_sdk-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


