# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/hice1/kshenoy8/scratch/stereosquad/openMVG_Build/_deps/rerun_sdk-src"
  "/home/hice1/kshenoy8/scratch/stereosquad/openMVG_Build/_deps/rerun_sdk-build"
  "/storage/ice1/3/7/kshenoy8/stereosquad/openMVG_Build/_deps/rerun_sdk-subbuild/rerun_sdk-populate-prefix"
  "/storage/ice1/3/7/kshenoy8/stereosquad/openMVG_Build/_deps/rerun_sdk-subbuild/rerun_sdk-populate-prefix/tmp"
  "/storage/ice1/3/7/kshenoy8/stereosquad/openMVG_Build/_deps/rerun_sdk-subbuild/rerun_sdk-populate-prefix/src/rerun_sdk-populate-stamp"
  "/storage/ice1/3/7/kshenoy8/stereosquad/openMVG_Build/_deps/rerun_sdk-subbuild/rerun_sdk-populate-prefix/src"
  "/storage/ice1/3/7/kshenoy8/stereosquad/openMVG_Build/_deps/rerun_sdk-subbuild/rerun_sdk-populate-prefix/src/rerun_sdk-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/storage/ice1/3/7/kshenoy8/stereosquad/openMVG_Build/_deps/rerun_sdk-subbuild/rerun_sdk-populate-prefix/src/rerun_sdk-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/storage/ice1/3/7/kshenoy8/stereosquad/openMVG_Build/_deps/rerun_sdk-subbuild/rerun_sdk-populate-prefix/src/rerun_sdk-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
