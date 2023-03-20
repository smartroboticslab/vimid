/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#include <kernels.h>
#include <default_parameters.h>
#include <vimidAction.h>

PerfStats Stats;

int main(int argc, char ** argv){
    Configuration config = parseArgs(argc, argv);
    vimidAction *vimid = new vimidAction(640, 480);
    vimid->init(&config);
    vimid->run();
    return 1;
}


