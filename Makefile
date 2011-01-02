################################################################################
#
#	Makefile
#  fastrack
#
#  Created by Dwight Bell on 12/13/10.
#  Copyright dbelll 2010. All rights reserved.
#

#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# This software and the information contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a Non-Disclosure Agreement.  Any reproduction or
# disclosure to any third party without the express written consent of
# NVIDIA is prohibited.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
#	In the project directory:
#
#		mkdir tmp
#		ln -s "$CUDASDK_HOME/lib" .
#		ln -s "$CUDASDK_HOME/common" .
#		ln -s "$CUDASDK_HOME/../shared" ../shared    (may already exist)
#	
################################################################################

# Add source files here
EXECUTABLE	:= fastrack

# Cuda source files (compiled with cudacc)
CUFILES		:= fastrack.cu cuda_utils.cu cuda_rand.cu

# CUDA dependency files
CU_DEPS	:= main.h helpers.h cuda_utils.h rand_utils.h fastrack.h dc_memory.h wgt_pointers.h board.h prototypes.h in_output.h results.h Makefile

# C/C++ source files (compiled with gcc / c++)
CCFILES		:=  main.c helpers.c rand_utils.c misc_utils.c

# additional includes
#INCLUDES = -I/home/dbelll/cuda_libraries/

# compiler flags
CUDACCFLAGS = --profile --ptxas-options=-v --use_fast_math #--maxrregcount=16

################################################################################
# Rules and targets
verbose ?= 1
ROOTDIR=tmp
include ./common/common.mk

